"""Format-v2 register allocator: width-aliased views, register classes and
pairs, early-clobber defs, exception edges, callee-save rules and explicit
per-edge parallel copies.

Bank model
==========
Every declared view starts at bit 0 of its bank in the shipped targets, so
two values can share physical bits at one program point exactly when their
placements use the same *bank* (a pair value occupies two banks at once).
Widths still matter in three places: (a) an operand may only use a view that
belongs to its register class; (b) a call clobbers the banks named by its
``clobbersv`` views and values survive only in callee-saved banks or on the
stack; (c) stack slots have a width-derived size and alignment.

Pipeline (rebuilt from scratch after every spill, so results are
deterministic)::

    liveness_v2 (normal + exception edges, per-edge parallel copies)
      -> per-point live sets and live intervals (global point numbering)
      -> interference graph over *bank sets* (one node = one value or one
         short reload/store/copy temporary; fixed views are precolored
         pseudo nodes; call clobbers are precolored pseudo nodes)
      -> greedy DSATUR placement into declared class options (bank sets),
         then bounded exact backtracking (proves impossibility or returns
         "budget")
      -> on impossibility, spill one victim and rebuild liveness + graph +
         intervals; stack slots are interval-packed with reuse and alignment
      -> lower2 materialises copies / loads / stores and breaks parallel
         copy cycles with a genuinely free register option or a private
         scratch stack slot (never a live aliased location)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .liveness2 import liveness_v2
from .liveness import predecessors
from .model import (
    BudgetExhaustedError,
    InfeasibleProgramError,
    Program,
    ProgramFormatError,
    RegAllocError,
)

COLORING_BUDGET_V2 = 200_000
# search stops being useful well before the graph can grow this large per
# attempt, but the cap keeps pathological inputs inside the demo budget.


@dataclass(frozen=True)
class EdgeCopySpec:
    dst: int
    src: int


@dataclass
class V2Plan:
    # var -> class name
    var_class: dict[int, str]
    # per-instruction operand handling (node ids are >= 0 real, < 0 pseudo)
    instr_pre: dict[int, list[tuple]] = field(default_factory=dict)
    instr_post: dict[int, list[tuple]] = field(default_factory=dict)
    instr_reads: dict[int, list[int]] = field(default_factory=dict)
    instr_writes: dict[int, list[int]] = field(default_factory=dict)
    # edge (p,s) -> list of (dst, src) parallel assignments (phis + explicit)
    edge_copies: dict[tuple[int, int], list[EdgeCopySpec]] = field(default_factory=dict)
    # edge (p,s) -> lowered records ("copy"/"load"/"store") with node ids
    edge_records: dict[tuple[int, int], list[tuple]] = field(default_factory=dict)
    adj: dict[int, set[int]] = field(default_factory=dict)
    precolor: dict[int, frozenset[int]] = field(default_factory=dict)
    real_nodes: set[int] = field(default_factory=set)
    temp_nodes: set[int] = field(default_factory=set)
    spilled: set[int] = field(default_factory=set)
    options: dict[int, tuple[frozenset[int], ...]] = field(default_factory=dict)
    # node id (real/temp/pseudo) -> register class name
    node_class: dict[int, str] = field(default_factory=dict)
    report: object = None
    # per-instruction per-operand chosen concrete views for fixedv
    loop_depth: dict[int, int] = field(default_factory=dict)


def _class_of_var(program: Program) -> dict[int, str]:
    rt = program.target.resolved()
    default = rt.default_class_name
    cls: dict[int, str] = {}
    for v in program.params:
        cls[v] = program.param_classes.get(v, default)
    for b in program.blocks:
        for ins in b.instrs:
            for v in ins.defs:
                cls[v] = ins.classes.get(v, default)
        for phi in b.phis:
            cls[phi.def_var] = default
        for ec in b.edge_copies:
            for d, _ in ec.pairs:
                cls.setdefault(d, default)
    return cls


def value_size(rt, class_name: str) -> int:
    """Stack units occupied by a value of ``class_name``.

    One unit is the default class width (one machine word).  Narrower values
    still take one unit (no sub-word packing of slots); a value wider than a
    word occupies ``ceil(width / word)`` units.
    """
    word = rt.class_info(rt.default_class_name).width
    w = rt.class_info(class_name).width
    return max(1, -(-w // word))  # ceil division


def _edge_parallel_copies(program: Program) -> dict[tuple[int, int], list[EdgeCopySpec]]:
    """Unify phi edge assignments and explicit edge_copies per edge."""
    out: dict[tuple[int, int], list[EdgeCopySpec]] = {}
    for s, b in enumerate(program.blocks):
        for phi in b.phis:
            for p, src in phi.sources:
                out.setdefault((p, s), []).append(EdgeCopySpec(dst=phi.def_var, src=src))
    for p, b in enumerate(program.blocks):
        for ec in b.edge_copies:
            for d, s in ec.pairs:
                out.setdefault((p, ec.dst_block), []).append(EdgeCopySpec(dst=d, src=s))
    return out


def _compute_loop_depths(program: Program) -> dict[int, int]:
    n = len(program.blocks)
    succs = [list(b.succs) for b in program.blocks]
    preds = predecessors(program.blocks)
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


def _occurrence_weights(program: Program) -> dict[int, int]:
    depth = _compute_loop_depths(program)
    w: dict[int, int] = {}

    def add(v, f):
        w[v] = w.get(v, 0) + f

    for bi, b in enumerate(program.blocks):
        f = 10 ** depth[bi]
        for ins in b.instrs:
            for v in ins.uses:
                add(v, f)
            for v in ins.defs:
                add(v, f)
        for phi in b.phis:
            add(phi.def_var, f)
            for _, s in phi.sources:
                add(s, f)
        for ec in b.edge_copies:
            for d, s in ec.pairs:
                add(d, f)
                add(s, f)
    return w


def build_v2(program: Program, spilled: set[int]) -> V2Plan:
    rt = program.target.resolved()
    report = liveness_v2(program)
    var_class = _class_of_var(program)
    edge_copies = _edge_parallel_copies(program)

    real_count = program.var_count()
    state = {"temp": real_count, "pseudo": -1}
    plan = V2Plan(var_class=var_class, spilled=set(spilled))
    adj: dict[int, set[int]] = {}
    plan.adj = adj
    plan.edge_copies = edge_copies

    def node(v):
        adj.setdefault(v, set())

    def new_temp():
        t = state["temp"]
        state["temp"] += 1
        adj[t] = set()
        plan.temp_nodes.add(t)
        return t

    def new_pseudo(banks: frozenset[int], cls: str):
        p = state["pseudo"]
        state["pseudo"] -= 1
        adj[p] = set()
        plan.precolor[p] = banks
        plan.node_class[p] = cls
        return p

    def edge(a, b):
        if a == b:
            return
        node(a)
        node(b)
        adj[a].add(b)
        adj[b].add(a)

    def clique(values):
        uniq = list(dict.fromkeys(values))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                edge(uniq[i], uniq[j])

    S = spilled

    # the set of variables that actually exist (ids may be sparse)
    existing: set[int] = set(program.params)
    for b in program.blocks:
        for ins in b.instrs:
            existing.update(ins.defs)
        for phi in b.phis:
            existing.add(phi.def_var)
        for ec in b.edge_copies:
            existing.update(d for d, _ in ec.pairs)
    plan.real_nodes = {v for v in existing if v not in S}

    def options_for(v):
        info = rt.class_info(var_class[v])
        return info.options

    for v in plan.real_nodes:
        node(v)
        plan.options[v] = options_for(v)
        plan.node_class[v] = var_class[v]

    S = spilled

    # ---- per-instruction operand plans ------------------------------------
    for bi, b in enumerate(program.blocks):
        for j, ins in enumerate(b.instrs):
            pre: list[tuple] = []
            post: list[tuple] = []
            reads: list[int] = []
            writes: list[int] = []
            # concrete nodes at each point
            read_concrete: list[int] = []
            write_concrete: list[int] = []
            # underlying real nodes that remain present through a fixed pin
            # (they must clash with everything except their own pseudo)
            read_extra: list[int] = []
            # one entry per def: [underlying real node] when a fixed,
            # unspilled def keeps its home value present, else []
            def_extra: list[list[int]] = [[] for _ in ins.defs]
            copy_pairs: set[tuple[int, int]] = set()
            survive = set(report.instr_after[bi][j]) - set(ins.defs)
            survive = {v for v in survive if v not in S}
            exc_live = {v for v in report.instr_except_live[bi][j] if v not in S}

            def join(group):
                uniq = list(dict.fromkeys(group))
                for x_i in range(len(uniq)):
                    for y_i in range(x_i + 1, len(uniq)):
                        x, y = uniq[x_i], uniq[y_i]
                        if (x, y) in copy_pairs or (y, x) in copy_pairs:
                            continue
                        edge(x, y)

            def operand_node(v: int, is_def: bool, def_k: int = -1):
                """Resolve the concrete node carrying operand v at the op."""
                cname = var_class[v]
                info = rt.class_info(cname)
                if v in ins.fixedv:
                    view_name = ins.fixedv[v]
                    bank = rt.view(view_name).bank
                    if info.is_pair:
                        opts = [opt for opt in info.options if bank in opt]
                        if len(opts) != 1:
                            raise ProgramFormatError(
                                f"blocks[{bi}].instrs[{j}].fixedv[{v}]: view {view_name!r} "
                                f"does not uniquely determine a pair option"
                            )
                        banks = opts[0]
                    else:
                        banks = frozenset({bank})
                    p = new_pseudo(banks, cname)
                    if is_def:
                        if v in S:
                            if v in report.instr_after[bi][j] or v in exc_live:
                                post.append(("store", p, v))
                        else:
                            post.append(("copy", v, p))
                            copy_pairs.add((v, p))
                            def_extra[def_k].append(v)
                    else:
                        if v in S:
                            t = new_temp()
                            plan.options[t] = info.options
                            plan.node_class[t] = cname
                            pre.append(("load", t, v))
                            pre.append(("copy", p, t))
                            copy_pairs.add((p, t))
                            # t physically holds the loaded value through the
                            # pre-copy and coexists with every survivor
                            read_concrete.append(t)
                            # a spilled value whose home is its slot must be
                            # written back when it survives this instruction:
                            # the fixed register only holds it for the op
                            if v in survive:
                                post.append(("store", p, v))
                        else:
                            pre.append(("copy", p, v))
                            copy_pairs.add((p, v))
                            read_extra.append(v)
                    return p
                if v in S:
                    t = new_temp()
                    plan.options[t] = info.options
                    plan.node_class[t] = cname
                    if is_def:
                        # store a spilled def only when it is live after this
                        # point; a dead-at-once result needs no slot and must
                        # not overwrite a slot reused by another live value
                        if v in report.instr_after[bi][j] or v in exc_live:
                            post.append(("store", t, v))
                    else:
                        pre.append(("load", t, v))
                    return t
                return v

            for u in ins.uses:
                nid = operand_node(u, False)
                reads.append(nid)
                read_concrete.append(nid)
            for dk, d in enumerate(ins.defs):
                nid = operand_node(d, True, dk)
                writes.append(nid)
                write_concrete.append(nid)

            plan.instr_pre[ins.uid] = pre
            plan.instr_post[ins.uid] = post
            plan.instr_reads[ins.uid] = reads
            plan.instr_writes[ins.uid] = writes

            # read point: concrete inputs and pinned source homes coexist
            # (except within a coalescing copy pair), with survivors and the
            # values required on an exception transfer
            join(read_concrete + read_extra + sorted(survive) + sorted(exc_live))
            early_idx = [k for k, d in enumerate(ins.defs) if d in ins.early_clobber]
            normal_idx = [k for k in range(len(ins.defs)) if k not in early_idx]
            early_nodes = [writes[k] for k in early_idx]
            early_nodes += [v for k in early_idx for v in def_extra[k]]
            normal_nodes = [writes[k] for k in normal_idx]
            normal_nodes += [v for k in normal_idx for v in def_extra[k]]
            # early-clobber defs are written BEFORE inputs are consumed
            if early_nodes:
                join(early_nodes + read_concrete + read_extra
                     + sorted(survive) + sorted(exc_live))
            # write point: defs pairwise + with survivors; a NORMAL def does
            # not conflict with a dying input
            join(early_nodes + normal_nodes + sorted(survive))

            # call clobbers: one precolored pseudo per clobbered bank; it
            # conflicts with every value that must still exist after/through
            # the call (normal survivors + exception-live values).  The call's
            # own inputs are consumed before the clobber and its results are
            # written afterwards, so operand pseudos may occupy those banks.
            clob_banks = {rt.view(vn).bank for vn in ins.clobbersv}
            for bank in sorted(clob_banks):
                p = new_pseudo(frozenset({bank}), rt.default_class_name)
                for v in sorted(survive) + sorted(exc_live):
                    edge(p, v)

    # ---- edge points: parallel copies + crossing values -------------------
    # Per-edge records consumed by the lowerer.  Each record is one of:
    #   ("copy",  dst_node, src_node)          register->register
    #   ("load",  dst_node, slot_dst_var)      slot -> register (dst spilled)
    #   ("store", src_node, slot_dst_var)      register -> slot (dst spilled)
    edge_records: dict[tuple[int, int], list[tuple]] = {}
    for (p, s), copies in edge_copies.items():
        dst_vars = {c.dst for c in copies}
        # values physically crossing the boundary INTO s: successor live-in
        # minus the values born on this edge (phi/edge-copy destinations).
        # Edge sources are NOT crossing values (they are consumed here).
        crossing = {
            v for v in report.live_in[s]
            if v not in S and v not in dst_vars
        }
        records: list[tuple] = []
        for c in copies:
            d, src = c.dst, c.src
            if d in S:
                if src in S:
                    t = new_temp()
                    plan.options[t] = rt.class_info(var_class[src]).options
                    plan.node_class[t] = var_class[src]
                    records.append(("load", t, src))
                    records.append(("store", t, d))
                else:
                    records.append(("store", src, d))
            else:
                if src in S:
                    t = new_temp()
                    plan.options[t] = rt.class_info(var_class[d]).options
                    plan.node_class[t] = var_class[d]
                    records.append(("load", t, src))
                    records.append(("copy", d, t))
                else:
                    records.append(("copy", d, src))
        edge_records[(p, s)] = records

        # ---- interference for the simultaneous read point -------------
        # The only registers that coexist at the read point are:
        #   * the distinct SOURCE values (read simultaneously), and
        #   * the crossing values live into the successor.
        # A source that dies on this edge frees its banks for the group's
        # destinations once the simultaneous read is over, so destinations
        # do NOT clash with a dying source unless that bank is still needed
        # by a crossing value or by another (surviving) source.  This is
        # exactly what makes a register-to-register permutation cycle
        # expressible at all; it is broken at lowering with a genuinely free
        # option or the private scratch slot.
        src_nodes: list[int] = []
        for rec in records:
            src_nodes.append(rec[2] if rec[0] == "copy" else rec[1])
        # a source survives the group when it also crosses into the successor
        # (a reload temp or a pure edge/phi source does not)
        crossing_set = set(crossing)
        surviving_sources = {x for x in src_nodes if x in crossing_set}

        def clash(a, b):
            edge(a, b)  # dying-source reuse is handled structurally below

        uniq_src = list(dict.fromkeys(src_nodes))
        # distinct simultaneous sources coexist with each other
        for i in range(len(uniq_src)):
            for j in range(i + 1, len(uniq_src)):
                edge(uniq_src[i], uniq_src[j])
        # surviving sources and crossing values occupy their banks through
        # the whole transfer and clash with every source and crossing value
        through = list(dict.fromkeys(list(surviving_sources) + sorted(crossing)))
        clique(through)
        for x in uniq_src:
            for e in through:
                clash(x, e)
        # destinations clash with everything that must STILL exist after the
        # read: crossing values and surviving sources.  They do NOT clash with
        # a dying source (its bank is reusable by the parallel assignment).
        dst_nodes = [rec[1] for rec in records if rec[0] in ("copy", "load")]
        for d in dict.fromkeys(dst_nodes):
            for e in through:
                clash(d, e)

    plan.edge_records = edge_records
    plan.report = report

    # ---- entry: parameters coexist ----------------------------------------
    entry = {v for v in program.params if v not in S}
    clique(sorted(entry))

    plan.loop_depth = _compute_loop_depths(program)
    return plan


# --------------------------------------------------------------------------
# coloring over bank sets
# --------------------------------------------------------------------------

def _precolor_conflicts(plan: V2Plan):
    for u, nbrs in plan.adj.items():
        if u in plan.precolor:
            for v in nbrs:
                if v in plan.precolor and plan.precolor[v] & plan.precolor[u] and v < u:
                    return (v, u, plan.precolor[u] & plan.precolor[v])
    return None


def _candidate_options(plan: V2Plan, v: int, color: dict[int, frozenset[int]]):
    if v in plan.precolor:
        return [plan.precolor[v]]
    used_banks: set[int] = set()
    for n in plan.adj[v]:
        if n in color:
            used_banks |= color[n]
    return [opt for opt in plan.options[v] if not (opt & used_banks)]


def greedy_color_v2(plan: V2Plan):
    color: dict[int, frozenset[int]] = dict(plan.precolor)
    remaining = set(plan.adj) - set(color)
    while remaining:
        best = None
        best_key = None
        for v in remaining:
            occ: set[int] = set()
            for u in plan.adj[v]:
                if u in color:
                    occ |= color[u]
            key = (len(occ), len(plan.adj[v]), -v)
            if best_key is None or key > best_key:
                best_key = key
                best = v
        opts = _candidate_options(plan, best, color)
        if not opts:
            return None
        color[best] = opts[0]
        remaining.discard(best)
    return color


class _Budget2(Exception):
    pass


def exact_color_v2(plan: V2Plan, budget: int = COLORING_BUDGET_V2):
    color: dict[int, frozenset[int]] = dict(plan.precolor)
    spent = [0]

    # fixed placements must be mutually bank-disjoint on adjacency
    for u, nbrs in plan.adj.items():
        if u in plan.precolor:
            for v in nbrs:
                if v in plan.precolor and plan.precolor[v] & plan.precolor[u]:
                    return "infeasible", None

    def choose(remaining):
        best = None
        best_key = None
        for v in remaining:
            occ: set[int] = set()
            for u in plan.adj[v]:
                if u in color:
                    occ |= color[u]
            key = (len(occ), len(plan.adj[v]), -v)
            if best_key is None or key > best_key:
                best_key = key
                best = v
        return best

    def rec(remaining):
        if not remaining:
            return True
        spent[0] += 1
        if spent[0] > budget:
            raise _Budget2()
        v = choose(remaining)
        for opt in _candidate_options(plan, v, color):
            color[v] = opt
            remaining.discard(v)
            if rec(remaining):
                return True
            remaining.add(v)
            del color[v]
        return False

    try:
        remaining = set(plan.adj) - set(color)
        if rec(remaining):
            return "found", color
        return "infeasible", None
    except _Budget2:
        return "budget", None


# --------------------------------------------------------------------------
# stack slot allocation with interval reuse and alignment
# --------------------------------------------------------------------------

def build_intervals(program: Program, plan: V2Plan, report, extra_points: dict[int, list] | None = None):
    """Sorted disjoint point segments per value (its live occupancy).

    ``extra_points`` maps a temp/var node -> additional numbered points it
    occupies (used for edge scratch slots / edge reloads).
    """
    segs: dict[int, list[tuple[int, int]]] = {}

    def add_point(v, p):
        segs.setdefault(v, [])
        lst = segs[v]
        if lst and p <= lst[-1][1] + 1:
            if p > lst[-1][1]:
                lst[-1] = (lst[-1][0], p)
        else:
            lst.append((p, p))

    for bi, b in enumerate(program.blocks):
        for j, ins in enumerate(b.instrs):
            p_before = report.points[("instr", bi, j)]
            p_after = report.points[("instr", bi, j + 1)]
            live = set(report.instr_before[bi][j]) | set(report.instr_except_live[bi][j])
            for v in live:
                add_point(v, p_before)
            for v in report.instr_after[bi][j]:
                add_point(v, p_after)
        for s in b.succs:
            if report.edge_is_exception[(bi, s)]:
                for j, ins in enumerate(b.instrs):
                    if s in ins.except_to:
                        p = report.points[("edge-exc", bi, s, j)]
                        for v in report.edge_live[(bi, s)]:
                            add_point(v, p)
            else:
                p = report.points[("edge-normal", bi, s)]
                for v in report.edge_live[(bi, s)]:
                    add_point(v, p)

    if extra_points:
        for v, pts in extra_points.items():
            for p in pts:
                add_point(v, p)

    for v in segs:
        segs[v].sort()
    return segs


def segments_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][1] < b[j][0]:
            i += 1
        elif b[j][1] < a[i][0]:
            j += 1
        else:
            return True
    return False


def pack_slots(program: Program, needs: dict[int, int], intervals, align: int):
    """Assign aligned, non-overlapping slot intervals.

    ``needs`` maps var -> size in stack units.  Variables with disjoint live
    intervals share a slot when sizes and alignment permit.  Returns
    ``{var: (start, size)}`` and the frame high-water mark.
    """
    order = sorted(needs, key=lambda v: (-needs[v], v))
    placement: dict[int, tuple[int, int]] = {}
    placed: list[tuple[int, list]] = []  # (var, segments) sharing tracking

    # bucket slots by size; each entry is (start, size, segs_of_occupant)
    slots: list[dict] = []

    def aligned(start: int, size: int) -> int:
        # a size-units interval must start at a multiple of min(align, size):
        # double-width intervals need even starts when align >= 2
        req = min(align, size) if align > 1 else 1
        if start % req:
            start += req - (start % req)
        return start

    high = 0
    for v in order:
        size = needs[v]
        segs = intervals.get(v, [(0, 0)])
        chosen = None
        for slot in slots:
            if slot["size"] < size:
                continue
            if any(segments_overlap(segs, occ) for occ in slot["occ"]):
                continue
            # reuse: occupant intervals are disjoint; slot start works for
            # the larger of the two sizes only when alignment still holds
            chosen = slot
            break
        if chosen is None:
            start = aligned(high, size)
            chosen = {"start": start, "size": size, "occ": []}
            slots.append(chosen)
            high = max(high, start + size)
        chosen["occ"].append(segs)
        placement[v] = (chosen["start"], size)
        placed.append((v, segs))
    frame = max(high, program.stack.min_slots)
    # frame itself aligned up to align
    if align > 1 and frame % align:
        frame += align - (frame % align)
    return placement, frame


# --------------------------------------------------------------------------
# spill victim
# --------------------------------------------------------------------------

def _spill_victim_v2(plan: V2Plan, weights: dict[int, int]) -> int:
    candidates = sorted(plan.real_nodes)
    if not candidates:
        raise InfeasibleProgramError(
            "no variables remain to spill but bank-set constraints are "
            "unsatisfiable: proven no allocation exists"
        )

    def score(v):
        w = weights.get(v, 1)
        return ((len(plan.adj[v]) + 1) / w, -v)

    return max(candidates, key=score)


@dataclass
class AllocationV2:
    program: Program
    assignment: dict[int, tuple[str, ...]]   # var -> concrete home view(s)
    assignment_banks: dict[int, frozenset[int]]
    slots: dict[int, tuple[int, int]]        # var -> (start, size)
    frame_slots: int
    allocated_program: object
    liveness: object
    iterations: int
    history: tuple
    scratch_slots: dict[tuple[int, int], tuple[int, int, str]]
    verification: object = None


class VerificationFailureError(RegAllocError):
    """The independent checker could not verify the produced allocation.

    Carries the checker report (edge failures are located precisely); no
    machine program is returned to the caller in this case.
    """

    def __init__(self, report):
        self.report = report
        super().__init__("allocation failed independent verification: "
                         + report.failure_text())


def allocate_v2(program: Program) -> AllocationV2:
    if not isinstance(program, Program) or not program.is_v2():
        raise ProgramFormatError("allocate_v2: expected a version=2 Program")
    rt = program.target.resolved()
    weights = _occurrence_weights(program)
    spilled: set[int] = set()
    history: list[tuple[int, ...]] = []

    color = None
    plan = None
    for iteration in range(program.var_count() + 1):
        history.append(tuple(sorted(spilled)))
        plan = build_v2(program, spilled)
        clash = _precolor_conflicts(plan)
        if clash is not None:
            a, b, banks = clash
            raise InfeasibleProgramError(
                f"two fixed-view requirements occupy overlapping banks "
                f"{sorted(banks)} at one instruction point (nodes {a}, {b}); "
                f"proven infeasible"
            )
        greedy = greedy_color_v2(plan)
        if greedy is not None:
            color = greedy
        else:
            status, found = exact_color_v2(plan, budget=COLORING_BUDGET_V2)
            if status == "found":
                color = found
            elif status == "budget":
                raise BudgetExhaustedError(
                    f"v2 bank-set search budget exhausted at iteration {iteration + 1} "
                    f"with spilled vars {sorted(spilled)}; feasibility unknown; "
                    f"no machine program emitted"
                )
            else:
                color = None
        if color is not None:
            break
        victim = _spill_victim_v2(plan, weights)
        spilled.add(victim)
    else:  # pragma: no cover
        raise RegAllocError("internal: v2 spill iteration did not converge")

    # concrete home views for real variables
    assignment_banks = {v: color[v] for v in plan.real_nodes}
    assignment: dict[int, tuple[str, ...]] = {}
    for v, banks in assignment_banks.items():
        assignment[v] = rt.home_view(plan.var_class[v], banks)

    # stack slots for spilled values (rebuilt from fresh intervals)
    report = liveness_v2(program)
    needs = {v: value_size(rt, plan.var_class[v]) for v in spilled}
    intervals = build_intervals(program, plan, report)
    placement, frame = pack_slots(program, needs, intervals, program.stack.align)

    from .lower2 import lower_v2
    allocated_program, scratch_slots, frame_final = lower_v2(
        program, plan, color, placement, frame
    )

    alloc = AllocationV2(
        program=program,
        assignment=dict(sorted(assignment.items())),
        assignment_banks={v: frozenset(b) for v, b in sorted(assignment_banks.items())},
        slots=dict(sorted(placement.items())),
        frame_slots=frame_final,
        allocated_program=allocated_program,
        liveness=report,
        iterations=iteration + 1,
        history=tuple(history),
        scratch_slots=scratch_slots,
    )

    # mandatory independent verification of EVERY edge + slot overlap +
    # bounded paired execution; failure raises with exact locations and no
    # machine program is handed back.
    from .checker2 import verify_allocation
    vrep = verify_allocation(program, alloc)
    alloc.verification = vrep
    if not vrep.ok:
        raise VerificationFailureError(vrep)
    return alloc
