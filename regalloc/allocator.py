"""Interference graph, coloring and iterative spilling.

Pipeline (rebuilt from scratch on every spill iteration, so results are
deterministic)::

    edge-sensitive liveness  (liveness.liveness_with_phis)
      -> build virtual program for the current spilled set S:
           * fixed operand requirements become short-lived precolored
             pseudo nodes joined to the instruction by COPY actions (so a
             fixed register is required at the instruction, not for the
             whole variable lifetime);
           * call clobbers become precolored pseudo nodes that interfere
             with every value surviving the call;
           * spilled variables disappear into memory slots; each use gets
             a reload temporary and each def a store temporary;
           * phi edge copies are explicit per-edge parallel-copy actions.
      -> graph coloring with K colors (greedy DSATUR, then an exact bounded
         DSATUR search that can *prove* impossibility)
      -> on proven impossibility spill one real variable and rebuild
      -> budget exhaustion is reported separately from proven infeasibility
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .liveness import liveness_with_phis, predecessors
from .model import (
    MAX_K,
    MIN_K,
    BudgetExhaustedError,
    InfeasibleProgramError,
    Program,
    ProgramFormatError,
    RegAllocError,
    check_int,
)

# Backtracking assignments tried before feasibility is declared "unknown".
COLORING_BUDGET = 100_000


# --------------------------------------------------------------------------
# plans describing one spill-form construction (replayed verbatim by lowerer)
# --------------------------------------------------------------------------

# actions are tuples:
#   ("load",  dst_node, slot_var)
#   ("store", src_node, slot_var)
#   ("copy",  dst_node, src_node)
@dataclass
class InstrPlan:
    uid: int
    pre: list[tuple] = field(default_factory=list)
    reads: list[int] = field(default_factory=list)
    writes: list[int] = field(default_factory=list)
    post: list[tuple] = field(default_factory=list)


@dataclass
class EdgePlan:
    pred: int
    succ: int
    loads: list[tuple] = field(default_factory=list)   # before copies
    copies: list[tuple] = field(default_factory=list)  # ("copy", d, s)
    stores: list[tuple] = field(default_factory=list)  # after copies


@dataclass
class VirtualProgram:
    adj: dict[int, set[int]]
    precolor: dict[int, int]
    instr_plans: dict[int, InstrPlan]
    edge_plans: dict[tuple[int, int], EdgePlan]
    real_nodes: set[int]
    temp_nodes: set[int]
    spilled: set[int]
    loop_depth: dict[int, int]
    # A spilled parameter is stack-passed: its value lives in its spill slot
    # from the entry point and never occupies a register there.


def _compute_loop_depths(program: Program) -> dict[int, int]:
    """Natural-loop nesting depth per block via dominators + back edges."""
    n = len(program.blocks)
    succs = [list(b.succs) for b in program.blocks]
    preds = predecessors(program.blocks)

    # dominators (iterative)
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
    # back edges: i -> h with h dominates i
    depth = {i: 0 for i in range(n)}
    headers: dict[int, list[int]] = {}
    for i in range(n):
        for h in succs[i]:
            if h in dom[i]:
                headers.setdefault(h, []).append(i)
    # natural loop body for header h with back-edge node t:
    # blocks from which t reaches h without going through h + h itself
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
    """Weighted def/use/phi occurrences (x10 per surrounding loop level)."""
    depth = _compute_loop_depths(program)
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


def build_virtual_program(program: Program, spilled: set[int]) -> VirtualProgram:
    """Construct nodes/edges/plans for one concrete spilled-set attempt."""
    blocks = program.blocks
    n = len(blocks)
    report = liveness_with_phis(program)
    preds = predecessors(blocks)

    phi_defs: list[dict[int, dict[int, int]]] = []
    # phi_src[b][def_var] = {pred: src}
    all_phi_defs: list[set[int]] = []
    for b in blocks:
        m: dict[int, dict[int, int]] = {}
        for phi in b.phis:
            m[phi.def_var] = dict(phi.sources)
        phi_defs.append(m)
        all_phi_defs.append(set(m))

    real_count = program.var_count()
    next_temp = [real_count]  # positive ids for spill reload/store temps
    next_pseudo = [-1]        # negative ids for fixed/clobber pseudos
    adj: dict[int, set[int]] = {}
    precolor: dict[int, int] = {}

    def node(v: int) -> None:
        adj.setdefault(v, set())

    def new_temp() -> int:
        t = next_temp[0]
        next_temp[0] += 1
        adj[t] = set()
        return t

    def new_pseudo(color: int) -> int:
        p = next_pseudo[0]
        next_pseudo[0] -= 1
        adj[p] = set()
        precolor[p] = color
        return p

    def edge(a: int, b: int) -> None:
        if a == b:
            return
        node(a)
        node(b)
        adj[a].add(b)
        adj[b].add(a)

    def clique(members: list[int]) -> None:
        uniq = list(dict.fromkeys(members))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                edge(uniq[i], uniq[j])

    S = spilled
    real_nodes = {v for v in range(real_count) if v not in S}
    for v in real_nodes:
        node(v)

    instr_plans: dict[int, InstrPlan] = {}
    edge_plans: dict[tuple[int, int], EdgePlan] = {}

    # ---- per-block body scan (backward from outgoing edge live sets) -----
    # Phi defs of THIS block are NOT seeded in: they are born at the entry
    # boundary (handled by the entry clique / edge plans below) and become
    # live inside the body purely through their uses, exactly as in the
    # liveness transfer. Seeding them here would spuriously extend their
    # ranges across the whole body.
    for bi, b in enumerate(blocks):
        live: set[int] = set()
        for s in b.succs:
            live |= set(report.edge_live[(bi, s)])
        live = {v for v in live if v not in S}

        plans_rev: list[InstrPlan] = []
        for ins in reversed(b.instrs):
            plan = InstrPlan(uid=ins.uid)
            surviving = [v for v in live if v not in ins.defs]
            read_nodes: list[int] = []   # values present while inputs are read
            write_nodes: list[int] = []  # values present while outputs are written
            copy_pairs: set[tuple[int, int]] = set()

            for u in ins.uses:
                if u in ins.fixed:
                    r = ins.fixed[u]
                    p = new_pseudo(r)
                    if u in S:
                        t = new_temp()
                        plan.pre.append(("load", t, u))
                        plan.pre.append(("copy", p, t))
                        copy_pairs.add((p, t))
                        read_nodes.append(t)
                    else:
                        plan.pre.append(("copy", p, u))
                        copy_pairs.add((p, u))
                        # u stays in its own register next to the pinned copy
                        read_nodes.append(u)
                    plan.reads.append(p)
                    read_nodes.append(p)
                elif u in S:
                    t = new_temp()
                    plan.pre.append(("load", t, u))
                    plan.reads.append(t)
                    read_nodes.append(t)
                else:
                    plan.reads.append(u)
                    read_nodes.append(u)

            for d in ins.defs:
                if d in ins.fixed:
                    r = ins.fixed[d]
                    p = new_pseudo(r)
                    if d in S:
                        plan.post.append(("store", p, d))
                    else:
                        plan.post.append(("copy", d, p))
                        copy_pairs.add((d, p))
                        # d holds the result after the op, coexisting with
                        # every value that survives the instruction
                        write_nodes.append(d)
                    plan.writes.append(p)
                    write_nodes.append(p)
                elif d in S:
                    t = new_temp()
                    plan.post.append(("store", t, d))
                    plan.writes.append(t)
                    write_nodes.append(t)
                else:
                    plan.writes.append(d)
                    write_nodes.append(d)

            # Standard three-address interference at this instruction:
            #   * all read operands coexist with each other and with every
            #     value that survives the instruction;
            #   * all written values coexist pairwise and with survivors;
            #   * a def does NOT interfere with a consumed use (the use dies
            #     at the instruction boundary).
            # Copy-related pairs are exempt (coalescing always allowed; the
            # lowerer emits a move when colors differ).
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

            # call clobbers: a pseudo node per clobbered register conflicts
            # only with values that survive the call
            for r in sorted(set(ins.clobbers)):
                c = new_pseudo(r)
                for v in surviving:
                    edge(c, v)

            plans_rev.append(plan)
            live = (live - set(ins.defs)) | set(ins.uses)
            live = {v for v in live if v not in S}

        for plan in reversed(plans_rev):
            instr_plans[plan.uid] = plan

    # ---- phi edge actions and interferences ------------------------------
    for bi, b in enumerate(blocks):
        if not b.phis:
            continue
        entry_values = [d for d in all_phi_defs[bi] if d not in S]
        crossing = [
            v
            for v in report.live_in[bi]
            if v not in all_phi_defs[bi] and v not in S
        ]
        clique(entry_values + crossing)

        for p in sorted(preds[bi]):
            ep = EdgePlan(pred=p, succ=bi)
            edge_plans[(p, bi)] = ep
            # Gather this edge's parallel assignment first, so copy coalescing
            # can be permitted only where it is provably safe.  Each record is
            # ("copy", dst_node, src_node) or ("store", src_node, slot_var).
            records: list[tuple] = []
            present: list[int] = []          # source-bearing nodes
            for phi in b.phis:
                d = phi.def_var
                s = dict(phi.sources)[p]
                if d in S:
                    if s in S:
                        t = new_temp()
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
                        t = new_temp()
                        ep.loads.append(("load", t, s))
                        ep.copies.append(("copy", d, t))
                        records.append(("copy", d, t))
                        present.append(t)
                    else:
                        ep.copies.append(("copy", d, s))
                        records.append(("copy", d, s))
                        present.append(s)

            # coalescing safety analysis for register copies.
            # copy record: ("copy", dst, src); store record: ("store", src, slot)
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
                # (1) d is not read by another action (true swap feeds a move)
                if read_count.get(d, 0) > 0:
                    continue
                # (2) s is not written by a different action (swap: s is a dest)
                if s in copy_dests and s != d:
                    continue
                # (3) s dies with this parallel group: a private reload temp
                #     always does; a real variable only if it neither crosses
                #     the boundary live nor is read by another action
                s_is_temp = s >= real_count
                if not s_is_temp:
                    if s in crossing_set:
                        continue
                    if read_count.get(s, 0) > 1:
                        continue
                safe.add((d, s))

            # sources coexist pairwise at the simultaneous read point
            uniq_present = list(dict.fromkeys(present))
            for i2 in range(len(uniq_present)):
                for j2 in range(i2 + 1, len(uniq_present)):
                    x, y = uniq_present[i2], uniq_present[j2]
                    if (x, y) in safe or (y, x) in safe:
                        continue
                    edge(x, y)
            # sources coexist with every value live at entry except the phi
            # definition that receives them via a coalescing-safe copy
            entries = entry_values + crossing
            for x in uniq_present:
                for e in entries:
                    if e == x or (x, e) in safe or (e, x) in safe:
                        continue
                    edge(x, e)

    # parameters all coexist at entry of block 0, except stack-passed
    # (spilled) ones, which live in their slot from the start
    entry0: set[int] = {v for v in program.params if v not in S}
    for v in report.live_in[0]:
        if v not in S:
            entry0.add(v)
    clique(sorted(entry0))

    temp_nodes = {v for v in adj if v >= real_count}
    return VirtualProgram(
        adj=adj,
        precolor=precolor,
        instr_plans=instr_plans,
        edge_plans=edge_plans,
        real_nodes=real_nodes,
        temp_nodes=temp_nodes,
        spilled=set(S),
        loop_depth=_compute_loop_depths(program),
    )


# --------------------------------------------------------------------------
# coloring
# --------------------------------------------------------------------------

def _precolor_conflicts(vp: VirtualProgram) -> tuple[int, int, int] | None:
    for u, vs in vp.adj.items():
        if u in vp.precolor:
            for v in vs:
                if v in vp.precolor and vp.precolor[v] == vp.precolor[u] and v < u:
                    return (v, u, vp.precolor[u])
    return None


def greedy_color(vp: VirtualProgram, k: int) -> dict[int, int] | None:
    """Deterministic DSATUR greedy.  Returns None if it gets stuck."""
    color: dict[int, int] = dict(vp.precolor)
    remaining = set(vp.adj) - set(color)
    while remaining:
        best = None
        best_key = None
        for v in remaining:
            sat = len({color[u] for u in vp.adj[v] if u in color})
            deg = len(vp.adj[v])
            key = (sat, deg, -v)  # max sat, max degree, min id
            if best_key is None or key > best_key:
                best_key = key
                best = v
        used = {color[u] for u in vp.adj[best] if u in color}
        pick = next((c for c in range(k) if c not in used), None)
        if pick is None:
            return None
        color[best] = pick
        remaining.discard(best)
    return color


def exact_color(vp: VirtualProgram, k: int, budget: int = COLORING_BUDGET) -> tuple[str, dict[int, int] | None]:
    """Bounded exact DSATUR search.

    Returns ``("found", coloring)``, ``("infeasible", None)`` (search tree
    exhausted -> proven no K-coloring) or ``("budget", None)``.
    """
    color = dict(vp.precolor)
    spent = [0]

    # static degree tie-break helper
    def choose(remaining: set[int]) -> int:
        best = None
        best_key = None
        for v in remaining:
            sat = len({color[u] for u in vp.adj[v] if u in color})
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
        used = {color[u] for u in vp.adj[v] if u in color}
        for c in range(k):
            if c in used:
                continue
            color[v] = c
            remaining.discard(v)
            if rec(remaining):
                return True
            remaining.add(v)
            del color[v]
        return False

    try:
        remaining = set(vp.adj) - set(color)
        # the fixed precoloration itself must be consistent: two adjacent
        # precolored nodes cannot share a colour
        for u, nbrs in vp.adj.items():
            cu = color.get(u)
            if cu is None:
                continue
            for v in nbrs:
                if color.get(v) == cu:
                    return "infeasible", None
        if rec(remaining):
            return "found", color
        return "infeasible", None
    except _Budget:
        return "budget", None


class _Budget(Exception):
    pass


# --------------------------------------------------------------------------
# spill selection / iteration
# --------------------------------------------------------------------------

def _spill_victim(vp: VirtualProgram, program: Program, weights: dict[int, int]) -> int:
    """Spill the variable with the highest (degree+1)/use-weight.

    High interference degree makes spilling it relieve the most pressure; a
    small weighted use count makes it cheap.  A spilled parameter is stored
    to its slot at entry, so parameters are spillable too.  Deterministic
    tie break: smaller variable id.
    """
    candidates = sorted(vp.real_nodes)
    if not candidates:
        raise InfeasibleProgramError(
            "no variables remain to spill but the coloring constraints are "
            "unsatisfiable: proven no allocation exists"
        )

    def score(v: int) -> tuple[float, int]:
        w = weights.get(v, 1)
        return ((len(vp.adj[v]) + 1) / w, -v)

    return max(candidates, key=score)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

@dataclass
class Allocation:
    program: Program
    k: int
    assignment: dict[int, int]
    slots: dict[int, int]
    allocated_program: "object"
    liveness: "object"
    iterations: int
    history: tuple[tuple[int, ...], ...]


def allocate(program: Program, k: int) -> Allocation:
    """Allocate ``program`` to ``k`` physical registers.

    Raises:
        ProgramFormatError: bad program/K (K outside 2..8, bool, NaN-like).
        InfeasibleProgramError: constraints proven contradictory.
        BudgetExhaustedError: coloring search budget exhausted (feasibility
            unknown).
    """
    if not isinstance(program, Program):
        raise ProgramFormatError("allocate: expected a Program instance")
    check_int(k, "K", minimum=MIN_K)
    if k > MAX_K:
        raise ProgramFormatError(f"K: expected integer <= {MAX_K}, got {k}")

    # every fixed/clobber register must fit K
    for bi, b in enumerate(program.blocks):
        for ii, ins in enumerate(b.instrs):
            for v, r in ins.fixed.items():
                if r >= k:
                    raise ProgramFormatError(
                        f"blocks[{bi}].instrs[{ii}].fixed[{v}]: "
                        f"register {r} does not exist for K={k}"
                    )
            for r in ins.clobbers:
                if r >= k:
                    raise ProgramFormatError(
                        f"blocks[{bi}].instrs[{ii}].clobbers: "
                        f"register {r} does not exist for K={k}"
                    )
    # Parameters exceeding K are feasible here: beyond the first K they are
    # treated as stack-passed (value lives in the spill slot at entry). The
    # intrinsic impossibility certificate below is independent of spilling.
    for bi, b in enumerate(program.blocks):
        for ii, ins in enumerate(b.instrs):
            nu = len(set(ins.uses))
            nd = len(set(ins.defs))
            if nu > k:
                raise InfeasibleProgramError(
                    f"blocks[{bi}].instrs[{ii}] (uid {ins.uid}) reads {nu} distinct "
                    f"operands at once but K={k}; spilling cannot remove a register "
                    f"need at the read point (proven infeasible)"
                )
            if nd > k:
                raise InfeasibleProgramError(
                    f"blocks[{bi}].instrs[{ii}] (uid {ins.uid}) writes {nd} distinct "
                    f"outputs at once but K={k} (proven infeasible)"
                )

    weights = _occurrence_weights(program)
    spilled: set[int] = set()
    history: list[tuple[int, ...]] = []
    budget_hit = False

    for iteration in range(program.var_count() + 1):
        history.append(tuple(sorted(spilled)))
        vp = build_virtual_program(program, spilled)

        clash = _precolor_conflicts(vp)
        if clash is not None:
            a, b, r = clash
            raise InfeasibleProgramError(
                f"two fixed-register requirements collide in register r{r} at an "
                f"instruction operand point (nodes {a}, {b}); proven infeasible"
            )

        greedy = greedy_color(vp, k)
        if greedy is not None:
            color = greedy
        else:
            status, found = exact_color(vp, k, budget=COLORING_BUDGET)
            if status == "found":
                color = found
            elif status == "budget":
                budget_hit = True
                color = None
            else:
                color = None

        if color is not None:
            # self-check: pseudos/temps always colored
            if any(v not in color for v in vp.adj):
                raise RegAllocError("internal: incomplete coloring")  # pragma: no cover
            break

        if budget_hit:
            raise BudgetExhaustedError(
                f"exact coloring search budget exhausted at iteration {iteration + 1} "
                f"with spilled vars {sorted(spilled)}; feasibility unknown"
            )

        victim = _spill_victim(vp, program, weights)
        spilled.add(victim)
    else:  # pragma: no cover
        raise RegAllocError("internal: spill iteration did not converge")

    assignment = {v: color[v] for v in vp.real_nodes if v in color}
    # slots are indexed by variable id, as required by the contract
    slots = {v: v for v in sorted(spilled)}

    # imported here to avoid a module cycle at import time
    from .lower import lower_program

    allocated_program = lower_program(program, vp, color)

    allocation = Allocation(
        program=program,
        k=k,
        assignment=dict(sorted(assignment.items())),
        slots=dict(sorted(slots.items())),
        allocated_program=allocated_program,
        liveness=liveness_with_phis(program),
        iterations=iteration + 1,
        history=tuple(history),
    )
    _post_validate(allocation, vp, color)
    return allocation


def _post_validate(allocation: Allocation, vp: VirtualProgram, color: dict[int, int]) -> None:
    """Defensive check of the promised invariant: two simultaneously live
    values never share a register (real variables only)."""
    p = allocation.program
    rep = allocation.liveness

    def reg_of(v: int) -> int | None:
        if v in allocation.slots:
            return None
        return allocation.assignment.get(v)

    # edge liveness uniqueness
    for (a, b), live in rep.edge_live.items():
        regs = {}
        for v in live:
            r = reg_of(v)
            if r is None:
                continue
            if r in regs and regs[r] != v:
                raise RegAllocError(  # pragma: no cover
                    f"internal: v{v} and v{regs[r]} share r{r} on edge {a}->{b}"
                )
            regs[r] = v
    # call clobbers: surviving real vars must avoid clobbered registers;
    # fixed requirements must be materialised by a copy (or already homed)
    for bi, blk in enumerate(p.blocks):
        for ii, ins in enumerate(blk.instrs):
            survive = rep.instr_after[bi][ii] - set(ins.defs)
            for v in survive:
                r = reg_of(v)
                if r is not None and r in ins.clobbers:
                    raise RegAllocError(  # pragma: no cover
                        f"internal: v{v} survives a call in clobbered r{r}"
                    )
            plan = vp.instr_plans[ins.uid]
            pairs = {(a, b) for kind, a, b in plan.pre + plan.post if kind == "copy"}
            for v, r in ins.fixed.items():
                if v in allocation.slots:
                    continue
                if allocation.assignment.get(v) == r:
                    continue
                if not any(src == v or dst == v for dst, src in pairs):
                    raise RegAllocError(  # pragma: no cover
                        f"internal: fixed requirement v{v}@r{r} not materialized"
                    )
