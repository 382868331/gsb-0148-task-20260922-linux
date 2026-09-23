"""V2 allocator: register views, aliases, exceptional edges, stack slots.

Pipeline (rebuilt from scratch on every spill iteration, so all derived
information — liveness, interference, slots — is recomputed after every
spill rewrite)::

    liveness_v2 (exceptional edges injected at the throw point)
      -> build virtual program for the current spilled set S:
           * each variable may only take a view of its own register class;
           * two simultaneously live values conflict unless their views
             share no bits (alias sets, not name inequality);
           * early-clobber defs conflict with the instruction's reads;
           * calls conflict survivors against every non-callee-saved view;
           * fixed_views become precolored pseudo nodes;
           * spilled variables disappear into stack slots (reload/store
             temporaries);
           * phi edge copies are per-edge parallel copies (normal and
             exceptional edges alike)
      -> view coloring (greedy DSATUR, then bounded exact DSATUR)
      -> on proven impossibility spill one variable and rebuild
      -> stack slots with reuse + alignment from the FINAL liveness
      -> budget exhaustion returns an explicit unallocated status

Unlike the v1 entry point, nothing here raises for the *outcome* of
allocation: the result object carries ``status`` =
``"allocated" | "infeasible" | "budget_exhausted"`` and, unless allocated,
no machine program is produced (``allocated_program is None``) — a budget
exhaustion can never masquerade as a finished program.  Malformed input
still raises :class:`ProgramFormatError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .liveness import liveness_v2, predecessors_v2, throw_index
from .machine import Machine
from .model import (
    Program,
    ProgramFormatError,
)

COLORING_BUDGET = 100_000

STATUS_ALLOCATED = "allocated"
STATUS_INFEASIBLE = "infeasible"
STATUS_BUDGET = "budget_exhausted"


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """One stack slot: byte ``offset``, ``size`` in bytes, ``align``."""

    offset: int
    size: int
    align: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class AllocationResult:
    """Outcome of :func:`allocate2`.

    ``assignment``/``slots``/``allocated_program`` are only meaningful when
    ``status == "allocated"``; on ``"infeasible"``/``"budget_exhausted"``
    they are ``None`` — an explicit unallocated state, never a
    seemingly-complete machine program.
    """

    status: str
    program: Program
    machine: Machine
    assignment: dict[int, str] | None
    slots: dict[int, Slot] | None
    scratch_slot: Slot | None
    frame_size: int | None
    allocated_program: object | None
    liveness: object | None
    iterations: int
    history: tuple[tuple[int, ...], ...]
    message: str = ""

    @property
    def allocated(self) -> bool:
        return self.status == STATUS_ALLOCATED


def _views_may_overlap(va, vb) -> bool:
    for a in va:
        for b in vb:
            if a.group == b.group and a.lo < b.hi and b.lo < a.hi:
                return True
    return False


def _classes_may_overlap(machine: Machine, a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """True iff some view of class/set A can share bits with some of B."""
    return _views_may_overlap(
        tuple(machine.view(v) for v in a), tuple(machine.view(v) for v in b)
    )


# --------------------------------------------------------------------------
# plans describing one spill-form construction (replayed by the lowerer)
# --------------------------------------------------------------------------


@dataclass
class InstrPlan2:
    uid: int
    pre: list[tuple] = field(default_factory=list)
    reads: list[int] = field(default_factory=list)
    writes: list[int] = field(default_factory=list)
    post: list[tuple] = field(default_factory=list)


@dataclass
class EdgePlan2:
    pred: int
    succ: int
    kind: str  # "normal" | "exc"
    loads: list[tuple] = field(default_factory=list)
    copies: list[tuple] = field(default_factory=list)
    stores: list[tuple] = field(default_factory=list)


@dataclass
class VirtualProgram2:
    adj: dict[int, set[int]]
    precolor: dict[int, str]
    allowed: dict[int, tuple[str, ...]]
    node_class: dict[int, str]
    instr_plans: dict[int, InstrPlan2]
    edge_plans: dict[tuple[int, int], EdgePlan2]
    real_nodes: set[int]
    temp_nodes: set[int]
    spilled: set[int]
    loop_depth: dict[int, int]


# --------------------------------------------------------------------------
# weights (loop-aware, deterministic)
# --------------------------------------------------------------------------


def _compute_loop_depths2(program: Program) -> dict[int, int]:
    """Natural-loop nesting depth per block (normal + exceptional edges)."""
    n = len(program.blocks)
    succs = [list(b.succs) + list(b.exc_succs) for b in program.blocks]
    preds = predecessors_v2(program.blocks)

    dom: list[set[int]] = [set(range(n)) for _ in range(n)]
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            new = {i} | (set.intersection(*(dom[p] for p in preds[i])) if preds[i] else set())
            if new != dom[i]:
                dom[i] = new
                changed = True
    depth = {i: 0 for i in range(n)}
    headers: dict[int, list[int]] = {}
    for i in range(n):
        for h in succs[i]:
            if h in dom[i]:
                headers.setdefault(h, []).append(i)
    for h, tails in headers.items():
        body = {h}
        stack = list(tails)
        while stack:
            x = stack.pop()
            if x in body:
                continue
            body.add(x)
            stack.extend(preds[x])
        for b in body:
            depth[b] += 1
    return depth


def _occurrence_weights2(program: Program) -> dict[int, int]:
    depth = _compute_loop_depths2(program)
    w: dict[int, int] = {}

    def add(v: int, factor: int) -> None:
        w[v] = w.get(v, 0) + factor

    for bi, b in enumerate(program.blocks):
        f = 10 ** depth[bi]
        for ins in b.instrs:
            for v in ins.uses:
                add(v, f)
            for v in ins.defs:
                add(v, f)
        for phi in b.phis:
            add(phi.def_var, f)
            for _, src in phi.sources:
                add(src, f)
    return w


# --------------------------------------------------------------------------
# virtual program construction
# --------------------------------------------------------------------------


def build_virtual_program2(
    program: Program, machine: Machine, spilled: set[int], report
) -> VirtualProgram2:
    """Construct nodes/edges/plans for one concrete spilled-set attempt."""
    blocks = program.blocks
    n = len(blocks)
    preds = predecessors_v2(blocks)

    var_classes = program.var_classes or {}

    def cls_of(v: int) -> str:
        return machine.class_of(v, var_classes)

    phi_defs: list[dict[int, dict[int, int]]] = []
    all_phi_defs: list[set[int]] = []
    for b in blocks:
        m: dict[int, dict[int, int]] = {}
        for phi in b.phis:
            m[phi.def_var] = dict(phi.sources)
        phi_defs.append(m)
        all_phi_defs.append(set(m))

    real_count = program.var_count()
    next_temp = [real_count]
    next_pseudo = [-1]
    adj: dict[int, set[int]] = {}
    precolor: dict[int, str] = {}
    allowed: dict[int, tuple[str, ...]] = {}
    node_class: dict[int, str] = {}

    def node(v: int) -> None:
        adj.setdefault(v, set())

    def new_temp(cls: str) -> int:
        t = next_temp[0]
        next_temp[0] += 1
        adj[t] = set()
        allowed[t] = machine.classes[cls]
        node_class[t] = cls
        return t

    def new_pseudo(view: str) -> int:
        p = next_pseudo[0]
        next_pseudo[0] -= 1
        adj[p] = set()
        precolor[p] = view
        allowed[p] = (view,)
        node_class[p] = machine.view(view).cls
        return p

    def edge(a: int, b: int) -> None:
        if a == b:
            return
        node(a)
        node(b)
        # Two nodes of different classes only conflict if SOME view
        # available to each can physically overlap.  A w-value living in
        # groups 0..2, for example, can never share bits with an s-value
        # living in groups 3..4, so no edge is needed (name inequality is
        # not the test — actual bit ranges are).
        if node_class[a] != node_class[b]:
            if not _classes_may_overlap(machine, allowed[a], allowed[b]):
                return
        adj[a].add(b)
        adj[b].add(a)

    def clique(members) -> None:
        uniq = list(dict.fromkeys(members))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                edge(uniq[i], uniq[j])

    S = spilled
    # real nodes are exactly the program's variables (ids may be sparse)
    all_vars = _all_vars(program)
    real_nodes = {v for v in all_vars if v not in S}
    for v in real_nodes:
        node(v)
        allowed[v] = machine.classes[cls_of(v)]
        node_class[v] = cls_of(v)

    instr_plans: dict[int, InstrPlan2] = {}
    edge_plans: dict[tuple[int, int], EdgePlan2] = {}

    # ---- per-instruction nodes and interferences -------------------------
    for bi, b in enumerate(blocks):
        for ii, ins in enumerate(b.instrs):
            plan = InstrPlan2(uid=ins.uid)
            surviving = sorted(
                v
                for v in report.instr_after[bi][ii]
                if v not in ins.defs and v not in S
            )
            read_nodes: list[int] = []
            write_nodes: list[int] = []
            copy_pairs: set[tuple[int, int]] = set()

            for u in ins.uses:
                if u in ins.fixed_views:
                    p = new_pseudo(ins.fixed_views[u])
                    if u in S:
                        t = new_temp(cls_of(u))
                        plan.pre.append(("load", t, u))
                        plan.pre.append(("copy", p, t))
                        copy_pairs.add((p, t))
                        read_nodes.append(t)
                    else:
                        plan.pre.append(("copy", p, u))
                        copy_pairs.add((p, u))
                        read_nodes.append(u)
                    plan.reads.append(p)
                    read_nodes.append(p)
                elif u in S:
                    t = new_temp(cls_of(u))
                    plan.pre.append(("load", t, u))
                    plan.reads.append(t)
                    read_nodes.append(t)
                else:
                    plan.reads.append(u)
                    read_nodes.append(u)

            def_write_node: dict[int, int] = {}
            for d in ins.defs:
                if d in ins.fixed_views:
                    p = new_pseudo(ins.fixed_views[d])
                    if d in S:
                        # a spilled fixed def is stored only when it lives
                        # past the instruction; otherwise the pinned value
                        # dies here and no slot write may happen (it could
                        # corrupt a reused slot holding a live value)
                        if d in report.instr_after[bi][ii]:
                            plan.post.append(("store", p, d))
                    else:
                        plan.post.append(("copy", d, p))
                        copy_pairs.add((d, p))
                        write_nodes.append(d)
                    plan.writes.append(p)
                    write_nodes.append(p)
                    def_write_node[d] = p
                elif d in S:
                    t = new_temp(cls_of(d))
                    if d in report.instr_after[bi][ii]:
                        plan.post.append(("store", t, d))
                    plan.writes.append(t)
                    write_nodes.append(t)
                    def_write_node[d] = t
                else:
                    plan.writes.append(d)
                    write_nodes.append(d)
                    def_write_node[d] = d

            def join(group: list[int]) -> None:
                uniq = list(dict.fromkeys(group))
                for i2 in range(len(uniq)):
                    for j2 in range(i2 + 1, len(uniq)):
                        x, y = uniq[i2], uniq[j2]
                        if (x, y) in copy_pairs or (y, x) in copy_pairs:
                            continue
                        edge(x, y)

            join(read_nodes + surviving)
            join(write_nodes + surviving)

            # early-clobber: the def is written before the reads finish, so
            # it conflicts with every value read by this instruction
            for d in ins.early_clobber:
                wn = def_write_node[d]
                for rn in read_nodes:
                    edge(wn, rn)

            # call/clobber: one precolored pseudo per destroyed view,
            # conflicting with every value that survives the instruction
            if ins.is_call:
                clobbered = machine.clobbered_by_call(ins.clobber_views)
            else:
                clobbered = tuple(dict.fromkeys(ins.clobber_views))
            for view in clobbered:
                c = new_pseudo(view)
                for v in surviving:
                    edge(c, v)

            instr_plans[ins.uid] = plan

    # ---- phi edge actions and interferences (normal + exceptional) -------
    for bi, b in enumerate(blocks):
        if not b.phis:
            continue
        # a phi result that is never read and never leaves the block is
        # dead: no edge actions are allowed for it (a store into its reused
        # slot could corrupt another value) and it gets no interference
        live_phi_defs: set[int] = set()
        for phi in b.phis:
            d = phi.def_var
            needed = d in report.live_out[bi] or any(
                d in row for row in report.instr_before[bi]
            )
            if needed:
                live_phi_defs.add(d)
        if not live_phi_defs:
            continue
        entry_values = [d for d in all_phi_defs[bi] if d in live_phi_defs and d not in S]
        crossing = sorted(
            v
            for v in report.live_in[bi]
            if v not in all_phi_defs[bi] and v not in S
        )
        clique(entry_values + crossing)

        for p in preds[bi]:
            kind = "exc" if bi in blocks[p].exc_succs else "normal"
            ep = EdgePlan2(pred=p, succ=bi, kind=kind)
            edge_plans[(p, bi)] = ep
            records: list[tuple] = []
            present: list[int] = []
            for phi in b.phis:
                d = phi.def_var
                if d not in live_phi_defs:
                    continue
                s = dict(phi.sources)[p]
                if d in S:
                    if s in S:
                        t = new_temp(cls_of(d))
                        ep.loads.append(("load", t, s))
                        ep.stores.append(("store", t, d))
                        records.append(("store", t, d))
                        present.append(t)
                    else:
                        ep.stores.append(("store", s, d))
                        records.append(("store", s, d))
                        present.append(s)
                else:
                    if s in S:
                        t = new_temp(cls_of(d))
                        ep.loads.append(("load", t, s))
                        ep.copies.append(("copy", d, t))
                        records.append(("copy", d, t))
                        present.append(t)
                    else:
                        ep.copies.append(("copy", d, s))
                        records.append(("copy", d, s))
                        present.append(s)

            # coalescing safety analysis (same rules as v1)
            read_count: dict[int, int] = {}
            copy_dests: set[int] = set()

            def rec_src(rec):
                return rec[2] if rec[0] == "copy" else rec[1]

            for rec in records:
                src = rec_src(rec)
                read_count[src] = read_count.get(src, 0) + 1
                if rec[0] == "copy":
                    copy_dests.add(rec[1])
            crossing_set = set(crossing)
            safe: set[tuple[int, int]] = set()
            for rec in records:
                if rec[0] != "copy":
                    continue
                d, s = rec[1], rec[2]
                if read_count.get(d, 0) > 0:
                    continue
                if s in copy_dests and s != d:
                    continue
                s_is_temp = s >= real_count
                if not s_is_temp:
                    if s in crossing_set:
                        continue
                    if read_count.get(s, 0) > 1:
                        continue
                safe.add((d, s))

            uniq_present = list(dict.fromkeys(present))
            for i2 in range(len(uniq_present)):
                for j2 in range(i2 + 1, len(uniq_present)):
                    x, y = uniq_present[i2], uniq_present[j2]
                    if (x, y) in safe or (y, x) in safe:
                        continue
                    edge(x, y)
            entries = entry_values + crossing
            for x in uniq_present:
                for e in entries:
                    if e == x or (x, e) in safe or (e, x) in safe:
                        continue
                    edge(x, e)

    # parameters coexist at entry (spilled ones are stack-passed)
    entry0: set[int] = {v for v in program.params if v not in S}
    for v in report.live_in[0]:
        if v not in S:
            entry0.add(v)
    clique(sorted(entry0))

    temp_nodes = {v for v in adj if v >= real_count}
    return VirtualProgram2(
        adj=adj,
        precolor=precolor,
        allowed=allowed,
        node_class=node_class,
        instr_plans=instr_plans,
        edge_plans=edge_plans,
        real_nodes=real_nodes,
        temp_nodes=temp_nodes,
        spilled=set(S),
        loop_depth=_compute_loop_depths2(program),
    )


# --------------------------------------------------------------------------
# view coloring
# --------------------------------------------------------------------------


def _forbidden(vp: VirtualProgram2, machine: Machine, color: dict[int, str], v: int) -> set[str]:
    out: set[str] = set()
    for u in vp.adj[v]:
        c = color.get(u)
        if c is not None:
            out |= machine.alias_closure[c]
    return out


def _precolor_clash(vp: VirtualProgram2, machine: Machine) -> tuple[int, int] | None:
    for u in sorted(vp.precolor):
        for v in vp.adj[u]:
            if v in vp.precolor and u < v:
                if machine.views_overlap(vp.precolor[u], vp.precolor[v]):
                    return (u, v)
    return None


def greedy_color2(vp: VirtualProgram2, machine: Machine) -> tuple[dict[int, str] | None, int | None]:
    """Deterministic DSATUR greedy over views.

    Returns ``(coloring, None)`` on success, or ``(None, stuck_node)`` —
    the node at which no view remained, used to target spill selection at
    the bottleneck.
    """
    color: dict[int, str] = dict(vp.precolor)
    remaining = set(vp.adj) - set(color)
    while remaining:
        best = None
        best_key = None
        for v in remaining:
            sat = len(_forbidden(vp, machine, color, v) & set(vp.allowed[v]))
            key = (sat, len(vp.adj[v]), -v)
            if best_key is None or key > best_key:
                best_key = key
                best = v
        forbidden = _forbidden(vp, machine, color, best)
        pick = next((vw for vw in vp.allowed[best] if vw not in forbidden), None)
        if pick is None:
            return None, best
        color[best] = pick
        remaining.discard(best)
    return color, None


class _Budget(Exception):
    pass


def exact_color2(
    vp: VirtualProgram2, machine: Machine, budget: int = COLORING_BUDGET
) -> tuple[str, dict[int, str] | None]:
    """Bounded exact DSATUR search over views.

    Returns ``("found", coloring)``, ``("infeasible", None)`` (proven no
    view assignment exists) or ``("budget", None)`` (unknown).
    """
    color: dict[int, str] = dict(vp.precolor)
    spent = [0]

    def choose(remaining: set[int]) -> int:
        best = None
        best_key = None
        for v in remaining:
            sat = len(_forbidden(vp, machine, color, v) & set(vp.allowed[v]))
            key = (sat, len(vp.adj[v]), -v)
            if best_key is None or key > best_key:
                best_key = key
                best = v
        return best

    def rec(remaining: set[int]) -> bool:
        if not remaining:
            return True
        spent[0] += 1
        if spent[0] > budget:
            raise _Budget()
        v = choose(remaining)
        forbidden = _forbidden(vp, machine, color, v)
        for vw in vp.allowed[v]:
            if vw in forbidden:
                continue
            color[v] = vw
            remaining.discard(v)
            if rec(remaining):
                return True
            remaining.add(v)
            del color[v]
        return False

    try:
        if _precolor_clash(vp, machine) is not None:
            return "infeasible", None
        remaining = set(vp.adj) - set(color)
        if rec(remaining):
            return "found", color
        return "infeasible", None
    except _Budget:
        return "budget", None


# --------------------------------------------------------------------------
# stack slots with reuse and alignment
# --------------------------------------------------------------------------


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def assign_slots(
    program: Program, machine: Machine, spilled: set[int], report
) -> tuple[dict[int, Slot], Slot, int]:
    """Assign byte offsets to spilled variables with reuse + alignment.

    Two spilled variables may share an offset iff they are never live at
    the same program point (checked against the FINAL liveness: every
    instruction before/after set, every block boundary and every edge,
    normal and exceptional).  Assignment is greedy in deterministic order
    (larger alignment, larger size, smaller id first) at the lowest aligned
    non-conflicting offset.  A dedicated scratch slot (used to break
    parallel-copy cycles) follows all variable slots at the strictest
    alignment; it never overlaps a variable slot.
    """
    var_classes = program.var_classes or {}

    def cls_of(v: int) -> str:
        return machine.class_of(v, var_classes)

    # collect co-liveness conflicts among spilled variables
    conflicts: dict[int, set[int]] = {v: set() for v in spilled}

    def note_point(live) -> None:
        here = [v for v in live if v in spilled]
        for i in range(len(here)):
            for j in range(i + 1, len(here)):
                conflicts[here[i]].add(here[j])
                conflicts[here[j]].add(here[i])

    for bi, b in enumerate(program.blocks):
        note_point(report.live_in[bi])
        for ii in range(len(b.instrs)):
            note_point(report.instr_before[bi][ii])
            note_point(report.instr_after[bi][ii])
        note_point(report.live_out[bi])
    for live in report.edge_live.values():
        note_point(live)
    note_point(set(program.params) | set(report.live_in[0]))

    order = sorted(
        spilled,
        key=lambda v: (
            -machine.class_align[cls_of(v)],
            -machine.class_width[cls_of(v)] // 8,
            v,
        ),
    )
    slots: dict[int, Slot] = {}
    for v in order:
        cls = cls_of(v)
        align = machine.class_align[cls]
        size = machine.class_width[cls] // 8
        off = 0
        bumped = True
        while bumped:
            bumped = False
            for u in sorted(conflicts[v] & set(slots), key=lambda x: slots[x].offset):
                su = slots[u]
                if off < su.end and su.offset < off + size:
                    off = _align_up(su.end, align)
                    bumped = True
        slots[v] = Slot(offset=off, size=size, align=align)

    max_align = max(machine.class_align.values())
    max_width = max(machine.class_width.values()) // 8
    top = 0
    for s in slots.values():
        top = max(top, s.end)
    scratch_off = _align_up(top, max_align)
    scratch = Slot(offset=scratch_off, size=max_width, align=max_align)
    frame = scratch.end
    return slots, scratch, frame


# --------------------------------------------------------------------------
# spill selection
# --------------------------------------------------------------------------


def _spill_victim2(
    vp: VirtualProgram2, weights: dict[int, int], bottleneck: int | None
) -> int | None:
    """Pick the variable to spill next.

    Candidates are the real-variable neighbours of the greedy bottleneck
    (the values actually competing at the failing point), falling back to
    all real variables.  Among candidates the *highest-degree* variable is
    chosen — spilling it relieves the most pressure at the bottleneck;
    lower weighted use count breaks ties (cheaper to keep in memory),
    then smaller id.  Deterministic.
    """
    candidates = sorted(vp.real_nodes)
    if not candidates:
        return None

    def score(v: int) -> tuple[int, int, int]:
        w = weights.get(v, 1)
        return (len(vp.adj[v]), -w, -v)

    if bottleneck is not None and bottleneck in vp.adj:
        bcls = vp.node_class.get(bottleneck)
        local = [v for v in vp.adj[bottleneck] if v in vp.real_nodes]
        # prefer competitors for the same register class — that is the
        # resource that ran out at the bottleneck
        same = [v for v in local if vp.node_class.get(v) == bcls]
        pool = same or local
        if pool:
            return max(pool, key=score)
    return max(candidates, key=score)


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _all_vars(program: Program) -> set[int]:
    out = set(program.params)
    for b in program.blocks:
        for ins in b.instrs:
            out.update(ins.defs)
            out.update(ins.uses)
        for phi in b.phis:
            out.add(phi.def_var)
            for _, src in phi.sources:
                out.add(src)
    return out


def _validate_v2(program: Program, machine: Machine) -> None:
    """Cross-field validation that needs the machine (format errors)."""
    if not isinstance(program, Program):
        raise ProgramFormatError("allocate2: expected a Program instance")
    if not isinstance(machine, Machine):
        raise ProgramFormatError("allocate2: expected a Machine instance")
    if program.var_classes is None:
        raise ProgramFormatError(
            "allocate2: legacy program (var_classes is None); use allocate()"
        )
    var_classes = program.var_classes
    for v, cls in sorted(var_classes.items()):
        if cls not in machine.classes:
            raise ProgramFormatError(
                f"var_classes[{v}]: class {cls!r} does not exist in machine "
                f"(have {sorted(machine.classes)})"
            )

    def cls_of(v: int) -> str:
        return machine.class_of(v, var_classes)

    for bi, b in enumerate(program.blocks):
        for ii, ins in enumerate(b.instrs):
            ip = f"blocks[{bi}].instrs[{ii}]"
            for v, view in ins.fixed_views.items():
                if view not in machine.view_map:
                    raise ProgramFormatError(
                        f"{ip}.fixed_views[{v}]: view {view!r} does not exist"
                    )
                want = cls_of(v)
                got = machine.view(view).cls
                if got != want:
                    raise ProgramFormatError(
                        f"{ip}.fixed_views[{v}]: view {view!r} is class {got!r} "
                        f"but v{v} is class {want!r}"
                    )
            for view in ins.clobber_views:
                if view not in machine.view_map:
                    raise ProgramFormatError(
                        f"{ip}.clobber_views: view {view!r} does not exist"
                    )
        for phi in b.phis:
            for p, src in phi.sources:
                if cls_of(src) != cls_of(phi.def_var):
                    raise ProgramFormatError(
                        f"blocks[{bi}].phis: phi v{phi.def_var} (class "
                        f"{cls_of(phi.def_var)!r}) takes source v{src} of "
                        f"class {cls_of(src)!r} on edge {p}->{bi}"
                    )

    # exceptional-edge well-formedness: a variable defined by the throwing
    # instruction or after it must not be live on an exceptional edge
    report = liveness_v2(program)
    for bi, b in enumerate(program.blocks):
        t = throw_index(b)
        if t is None:
            continue
        late_defs: set[int] = set()
        for ins in b.instrs[t:]:
            late_defs.update(ins.defs)
        for q in b.exc_succs:
            bad = sorted(late_defs & set(report.edge_live[(bi, q)]))
            if bad:
                raise ProgramFormatError(
                    f"blocks[{bi}]: variable(s) {bad} defined at or after the "
                    f"may_throw instruction are live on exceptional edge "
                    f"{bi}->{q}; they do not exist on that path"
                )


def _infeasible_certificates(
    program: Program, machine: Machine
) -> str | None:
    """Cheap PROVEN-infeasibility certificates (None = nothing proven)."""
    var_classes = program.var_classes or {}

    def cls_of(v: int) -> str:
        return machine.class_of(v, var_classes)

    for bi, b in enumerate(program.blocks):
        for ii, ins in enumerate(b.instrs):
            ip = f"blocks[{bi}].instrs[{ii}] (uid {ins.uid})"
            # pinned operands whose views overlap
            pinned = sorted(ins.fixed_views.items())
            early = set(ins.early_clobber)
            uses = set(ins.uses)
            for a in range(len(pinned)):
                for c in range(a + 1, len(pinned)):
                    va, ra = pinned[a]
                    vc, rc = pinned[c]
                    both_uses = va in uses and vc in uses
                    both_defs = va not in uses and vc not in uses
                    early_pair = (va in early) != (vc in early) or (
                        va in early and vc in early
                    )
                    if not (both_uses or both_defs or early_pair):
                        continue
                    if machine.views_overlap(ra, rc):
                        return (
                            f"{ip}: operands v{va} and v{vc} are pinned to "
                            f"overlapping views {ra!r}/{rc!r} (proven infeasible)"
                        )
            # pigeonhole per class at the read/write point
            for group, operands in (("reads", ins.uses), ("writes", ins.defs)):
                need: dict[str, int] = {}
                for v in operands:
                    c = cls_of(v)
                    need[c] = need.get(c, 0) + 1
                for c, cnt in sorted(need.items()):
                    have = len(machine.classes[c])
                    if cnt > have:
                        return (
                            f"{ip}: {cnt} distinct {group} of class {c!r} at "
                            f"one point but the machine has only {have} "
                            f"{c!r}-class view(s); spilling cannot remove a "
                            f"register need at the point itself "
                            f"(proven infeasible)"
                        )
    return None


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def allocate2(
    program: Program, machine: Machine, *, coloring_budget: int = COLORING_BUDGET
) -> AllocationResult:
    """Allocate a v2 ``program`` to the views of ``machine``.

    Never raises for the allocation *outcome*; returns an
    :class:`AllocationResult` whose ``status`` is one of
    ``"allocated"`` / ``"infeasible"`` / ``"budget_exhausted"``.
    Raises :class:`ProgramFormatError` for malformed input only.
    """
    _validate_v2(program, machine)

    def unallocated(status: str, message: str, report=None, history=(), it=0):
        return AllocationResult(
            status=status,
            program=program,
            machine=machine,
            assignment=None,
            slots=None,
            scratch_slot=None,
            frame_size=None,
            allocated_program=None,
            liveness=report,
            iterations=it,
            history=tuple(history),
            message=message,
        )

    certificate = _infeasible_certificates(program, machine)
    if certificate is not None:
        return unallocated(STATUS_INFEASIBLE, certificate)

    weights = _occurrence_weights2(program)
    spilled: set[int] = set()
    history: list[tuple[int, ...]] = []

    for iteration in range(program.var_count() + 1):
        history.append(tuple(sorted(spilled)))
        report = liveness_v2(program)
        vp = build_virtual_program2(program, machine, spilled, report)

        clash = _precolor_clash(vp, machine)
        if clash is not None:
            a, b = clash
            return unallocated(
                STATUS_INFEASIBLE,
                f"two fixed/clobber requirements collide in aliasing views "
                f"{vp.precolor[a]!r}/{vp.precolor[b]!r} (nodes {a}, {b}); "
                f"proven infeasible",
                report,
                history,
                iteration + 1,
            )

        color, stuck = greedy_color2(vp, machine)
        bottleneck = stuck
        if color is None:
            status, found = exact_color2(vp, machine, budget=coloring_budget)
            if status == "found":
                color = found
            elif status == "budget":
                return unallocated(
                    STATUS_BUDGET,
                    f"exact view-coloring search budget exhausted at iteration "
                    f"{iteration + 1} with spilled vars {sorted(spilled)}; "
                    f"feasibility unknown",
                    report,
                    history,
                    iteration + 1,
                )

        if color is not None:
            break

        victim = _spill_victim2(vp, weights, bottleneck)
        if victim is None:
            return unallocated(
                STATUS_INFEASIBLE,
                "no variables remain to spill but the view-coloring "
                "constraints are unsatisfiable: proven no allocation exists",
                report,
                history,
                iteration + 1,
            )
        spilled.add(victim)
    else:  # pragma: no cover
        raise ProgramFormatError("internal: spill iteration did not converge")

    slots, scratch, frame = assign_slots(program, machine, spilled, report)

    from .lower2 import lower_program2

    allocated_program = lower_program2(program, machine, vp, color, report, slots, scratch)

    assignment = {v: color[v] for v in sorted(vp.real_nodes) if v in color}
    return AllocationResult(
        status=STATUS_ALLOCATED,
        program=program,
        machine=machine,
        assignment=assignment,
        slots={v: slots[v] for v in sorted(slots)},
        scratch_slot=scratch,
        frame_size=frame,
        allocated_program=allocated_program,
        liveness=report,
        iterations=iteration + 1,
        history=tuple(history),
        message="",
    )
