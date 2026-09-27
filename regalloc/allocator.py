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
#   ("remat", dst_node, def_uid, operand_nodes)
@dataclass
class InstrPlan:
    uid: int
    pre: list[tuple] = field(default_factory=list)
    reads: list[int] = field(default_factory=list)
    writes: list[int] = field(default_factory=list)
    post: list[tuple] = field(default_factory=list)
    elided: bool = False
    remat_groups: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)


@dataclass
class EdgePlan:
    pred: int
    succ: int
    loads: list[tuple] = field(default_factory=list)   # before copies
    remats: list[tuple] = field(default_factory=list)  # rematerialisations
    copies: list[tuple] = field(default_factory=list)  # ("copy", d, s)
    stores: list[tuple] = field(default_factory=list)  # after copies
    remat_groups: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)


@dataclass
class VirtualProgram:
    adj: dict[int, set[int]]
    precolor: dict[int, int]
    instr_plans: dict[int, InstrPlan]
    edge_plans: dict[tuple[int, int], EdgePlan]
    real_nodes: set[int]
    temp_nodes: set[int]
    spilled: set[int]
    rematerialized: set[int]
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


def _use_weights(program: Program) -> dict[int, int]:
    """Weighted *use* occurrences only (x10 per surrounding loop level).

    Only uses (instruction uses and phi-source occurrences) are counted;
    definitions are not.  This weights the runtime cost of re-materialising
    a value (one recomputation per use) versus spilling it (a store at the
    definition plus a reload per use).
    """
    depth = _compute_loop_depths(program)
    w: dict[int, int] = {}

    def add(v: int, factor: int) -> None:
        w[v] = w.get(v, 0) + factor

    for bi, b in enumerate(program.blocks):
        f = 10 ** depth[bi]
        for ins in b.instrs:
            for v in ins.uses:
                add(v, f)
        for phi in b.phis:
            for _, src in phi.sources:
                add(src, f)
    return w


@dataclass(frozen=True)
class RematInfo:
    """Caller-supplied rematerialisation policy, after strict validation.

    ``costs`` maps instruction uid -> recomputation cost (positive finite
    number).  ``eligible`` maps a defined variable to the uid that can
    recompute it whenever every use is served: only single-definition,
    single-result, non-clobbering, no-fixed-register pure operations whose
    operands are themselves available at every use point are eligible.
    The marker is permission, not a command: an ineligible marker is simply
    never selected; structurally invalid markers raise ProgramFormatError.
    """

    costs: dict[int, float]
    eligible: dict[int, int]
    operands: dict[int, tuple[int, ...]]
    consumption: dict[int, tuple[tuple[int, object, tuple[int, ...]], ...]]
    edge_sources: dict[tuple[int, int], tuple[int, ...]]


def analyze_remat(program: Program, remat: dict[int, float]) -> RematInfo:
    """Validate ``{uid: cost}`` markers and compute static eligibility.

    Structural violations (non-pure operations: multiple results, any call
    clobber, fixed-register requirements; parameters; phi results; redefined
    variables; bad uid/cost) are format errors with a locatable path.
    A marker that is structurally fine but whose operands cannot be proven
    available at every consumption point is kept in ``costs`` but omitted
    from ``eligible`` (the marker is then simply never chosen).
    """
    if not isinstance(remat, dict):
        raise ProgramFormatError("remat: expected a mapping {uid: cost}")

    instr_by_uid: dict[int, tuple[int, int]] = {}  # uid -> (block, index)
    def_points: dict[int, list[int]] = {}          # var -> uids defining it
    instr_of_uid: dict[int, object] = {}
    for bi, b in enumerate(program.blocks):
        for ii, ins in enumerate(b.instrs):
            if ins.uid in instr_by_uid:
                raise ProgramFormatError(
                    f"blocks[{bi}].instrs[{ii}]: duplicate uid {ins.uid}"
                )
            instr_by_uid[ins.uid] = (bi, ii)
            instr_of_uid[ins.uid] = ins
            for d in ins.defs:
                def_points.setdefault(d, []).append(ins.uid)

    costs: dict[int, float] = {}
    legal: dict[int, int] = {}
    for raw_uid, raw_cost in remat.items():
        if isinstance(raw_uid, bool) or not isinstance(raw_uid, int):
            raise ProgramFormatError(
                f"remat key: expected instruction uid int, got {type(raw_uid).__name__}"
            )
        path = f"remat[{raw_uid}]"
        if raw_uid not in instr_by_uid:
            raise ProgramFormatError(f"{path}: no instruction with uid {raw_uid}")
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise ProgramFormatError(
                f"{path}: cost must be a positive finite number, got {type(raw_cost).__name__}"
            )
        cost = float(raw_cost)
        if cost <= 0 or cost != cost or cost in (float("inf"), float("-inf")):
            raise ProgramFormatError(
                f"{path}: cost must be a positive finite number, got {raw_cost!r}"
            )
        costs[raw_uid] = cost

        bi, ii = instr_by_uid[raw_uid]
        ins = program.blocks[bi].instrs[ii]
        why = None
        if len(ins.defs) != 1:
            why = ("rematerialisation is only allowed for a single pure result; "
                   f"the instruction defines {len(ins.defs)} variables")
        elif ins.clobbers:
            why = "an instruction that clobbers registers is not pure (possible call side effect)"
        elif ins.fixed:
            why = "an instruction with fixed-register requirements cannot be rematerialised"
        if why is not None:
            raise ProgramFormatError(
                f"blocks[{bi}].instrs[{ii}] (uid {ins.uid}): {why}"
            )
        d = ins.defs[0]
        if d in program.params:
            raise ProgramFormatError(f"{path}: a parameter cannot be rematerialised")
        if any(d == phi.def_var for b in program.blocks for phi in b.phis):
            raise ProgramFormatError(f"{path}: a phi result cannot be rematerialised")
        if len(def_points[d]) != 1:
            raise ProgramFormatError(
                f"{path}: variable v{d} is defined more than once; a rematerialised value "
                f"must have a single pure definition"
            )
        legal[d] = raw_uid

    # Fixed point: a candidate is usable at all its consumption points only
    # when every operand is, at that point, either (a) live with its unique
    # definition dominating the point (plain SSA availability), or (b) itself
    # a legal rematerialisation candidate.  Spilled operands are always
    # available (their home is a slot), and a value that is live-in at the
    # use block is available everywhere in that block.
    report = liveness_with_phis(program)
    candidates = set(legal)

    def unique_def_var(v: int) -> bool:
        if v in program.params:
            return True
        if len(def_points.get(v, ())) == 1:
            return True
        # a phi result is a single static definition too: at any dominated
        # use its value is exactly the source carried on the incoming edge
        return any(v == phi.def_var for b in program.blocks for phi in b.phis)

    # Collect the concrete consumption points of each candidate variable.
    # An instruction use is located at its instruction (operands = the use
    # list of that consuming instruction); a phi-source use is located at
    # the end of the predecessor block that supplies it (the edge actions
    # run there), with the operands of the candidate's own definition.
    uid_by_var = {d: uid for d, uid in legal.items()}
    consumption: dict[int, list[tuple[int, object, tuple[int, ...]]]] = {}
    edge_sources: dict[tuple[int, int], list[int]] = {}
    for bi2, b2 in enumerate(program.blocks):
        for ii2, ins2 in enumerate(b2.instrs):
            for u in ins2.uses:
                if u in uid_by_var:
                    d_ins = instr_of_uid[uid_by_var[u]]
                    before = report.instr_before[bi2][ii2]
                    consumption.setdefault(u, []).append(
                        (bi2, ii2, tuple(d_ins.uses), frozenset(before)))
        for phi in b2.phis:
            for pred, src in phi.sources:
                if src in uid_by_var:
                    d_ins = instr_of_uid[uid_by_var[src]]
                    edge_sources.setdefault((pred, bi2), []).append(src)
                    consumption.setdefault(src, []).append(
                        (pred, None, tuple(d_ins.uses),
                         frozenset(report.edge_live[(pred, bi2)])))

    eligible: set[int] = set()
    changed = True
    while changed:
        changed = False
        for d in candidates - eligible:
            ok = True
            for _bi, _ii, operands, live_here in consumption.get(d, ()):
                for u in operands:
                    if u == d or u in eligible:
                        continue
                    if u in candidates:
                        ok = False  # wait for its fixed-point decision
                        break
                    if not unique_def_var(u):
                        ok = False
                        break
                    if u not in live_here:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                eligible.add(d)
                changed = True

    return RematInfo(
        costs=dict(sorted(costs.items())),
        eligible={d: legal[d] for d in sorted(eligible)},
        operands={d: instr_of_uid[uid].uses for d, uid in legal.items()},
        consumption={d: tuple(v) for d, v in sorted(consumption.items())},
        edge_sources={e: tuple(vs) for e, vs in sorted(edge_sources.items())},
    )


def build_virtual_program(program: Program, spilled: set[int],
                          rematerialized: set[int] | None = None,
                          remat_info: RematInfo | None = None) -> VirtualProgram:
    """Construct nodes/edges/plans for one concrete spilled-set attempt."""
    blocks = program.blocks
    n = len(blocks)
    report = liveness_with_phis(program)
    preds = predecessors(blocks)
    S = spilled
    R = set(rematerialized or ())
    if R & S:
        raise RegAllocError("internal: a variable cannot be both spilled and rematerialised")

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

    def_info = remat_info
    def remat_chain(var: int, plan_holder, target_list) -> int:
        """Materialise ``var`` in a fresh temp at the use point.

        Spilled operands are loaded from their slot; rematerialised
        operands are recursively recomputed (each recomputation is its own
        short-lived temp + one replay of the pure op); any other operand is
        referenced directly.  Returns the graph node holding the value.
        """
        if var in S:
            t = new_temp()
            target_list.append(("load", t, var))
            return t
        if var in R:
            uid = def_info.eligible[var]
            op_nodes = [remat_chain(u, plan_holder, target_list)
                        for u in def_info.operands[var]]
            t = new_temp()
            target_list.append(("remat", t, uid, tuple(op_nodes)))
            plan_holder.remat_groups.append((t, op_nodes))
            return t
        return var

    real_nodes = {v for v in range(real_count) if v not in S and v not in R}
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
        live = {v for v in live if v not in S and v not in R}

        plans_rev: list[InstrPlan] = []
        for ins in reversed(b.instrs):
            plan = InstrPlan(uid=ins.uid)
            # If this instruction itself defines a rematerialised value, no
            # register-level op is emitted at the definition point: the pure
            # op is replayed only at its consumption points.  Its uses still
            # participate in ordinary liveness (they may be live for other
            # reasons), but the instruction writes nothing here.
            elided_defs = [d for d in ins.defs if d in R]
            if elided_defs:
                if len(ins.defs) != 1 or ins.defs[0] != elided_defs[0]:
                    raise RegAllocError(  # pragma: no cover
                        "internal: cannot elide part of a multi-def instruction"
                    )
                plan.elided = True
            surviving = [v for v in live if v not in ins.defs]
            read_nodes: list[int] = []   # values present while inputs are read
            write_nodes: list[int] = []  # values present while outputs are written
            copy_pairs: set[tuple[int, int]] = set()

            if not plan.elided:
                for u in ins.uses:
                    if u in ins.fixed:
                        r = ins.fixed[u]
                        p = new_pseudo(r)
                        if u in S or u in R:
                            t = remat_chain(u, plan, plan.pre)
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
                    elif u in S or u in R:
                        t = remat_chain(u, plan, plan.pre)
                        plan.reads.append(t)
                        read_nodes.append(t)
                    else:
                        plan.reads.append(u)
                        read_nodes.append(u)

            if not plan.elided:
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

            # Rematerialised operand temps are short-lived but must not
            # share a register with the other operands/survivors alive at
            # the replay point (no unallocatable extra pressure is assumed:
            # the graph builder rejects what K cannot hold like any node).
            for t, op_nodes in plan.remat_groups:
                clique([t] + list(op_nodes) + read_nodes + surviving)

            # call clobbers: a pseudo node per clobbered register conflicts
            # only with values that survive the call
            for r in sorted(set(ins.clobbers)):
                c = new_pseudo(r)
                for v in surviving:
                    edge(c, v)

            plans_rev.append(plan)
            live = (live - set(ins.defs)) | set(ins.uses)
            live = {v for v in live if v not in S and v not in R}

        for plan in reversed(plans_rev):
            instr_plans[plan.uid] = plan

    # ---- phi edge actions and interferences ------------------------------
    for bi, b in enumerate(blocks):
        if not b.phis:
            continue
        # A phi definition is a real register value at block entry even when
        # marked rematerialisable (the marker only permits elision; the phi
        # itself materialises it here), so it stays in the entry clique.
        entry_values = [d for d in all_phi_defs[bi] if d not in S]
        crossing = [
            v
            for v in report.live_in[bi]
            if v not in all_phi_defs[bi] and v not in S and v not in R
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
                s_node = remat_chain(s, ep, ep.remats)
                present.append(s_node)
                if d in S:
                    ep.stores.append(("store", s_node, d))
                    records.append(("store", s_node, d))
                else:
                    ep.copies.append(("copy", d, s_node))
                    records.append(("copy", d, s_node))

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
            # rematerialised phi-source temps coexist with each other and
            # with every value live across the boundary at the replay point
            for t, op_nodes in ep.remat_groups:
                clique([t] + list(op_nodes) + uniq_present + entries)

    # parameters all coexist at entry of block 0, except stack-passed
    # (spilled) ones, which live in their slot from the start
    entry0: set[int] = {v for v in program.params if v not in S}
    for v in report.live_in[0]:
        if v not in S and v not in R:
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
        rematerialized=set(R),
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

def _remat_closure(var: int, info: RematInfo) -> list[int]:
    """All eligible rematerialised variables ``var``'s replay recursively
    needs (itself included), smallest id first for determinism."""
    out: set[int] = set()
    stack = [var]
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        for u in info.operands.get(x, ()):
            if u in info.eligible:
                stack.append(u)
    return sorted(out)


def _choose_relief_action(vp: VirtualProgram, program: Program,
                          weights: dict[int, int], use_weights: dict[int, int],
                          spilled: set[int], remat_set: set[int],
                          info: RematInfo, k: int,
                          banned: frozenset[int] = frozenset()
                          ) -> tuple[str, int] | None:
    """Pick the best pressure-relief action: spill one variable or
    rematerialise one marked pure value (plus its dependency closure).

    Both alternatives are scored by *graph relief per runtime cost*:
    ``(degree + 1) / weighted_cost``.  Spill cost is the weighted occurrence
    weight (store at def + reloads at uses); rematerialisation cost is the
    caller-given recomputation cost times the weighted number of use points
    (the definition is elided, so there is no store).  Deterministic tie
    break: larger ratio, then prefer rematerialising, then smaller variable
    id.
    """
    choices: list[tuple[float, int, int, str, int]] = []
    for v in sorted(vp.real_nodes):
        relief = len(vp.adj[v]) + 1
        spill_cost = max(weights.get(v, 1), 1)
        choices.append((relief / spill_cost, 0, -v, "spill", v))
        if (v in info.eligible and v not in banned and v not in remat_set
                and _remat_intrinsic_feasible(
                    v, program, spilled | set(_remat_closure(v, info)), info, k)):
            remat_cost = max(use_weights.get(v, 1), 1) * info.costs[info.eligible[v]]
            choices.append((relief / remat_cost, 1, -v, "remat", v))
    if not choices:
        return None
    _, _, _, kind, v = max(choices, key=lambda c: (c[0], c[1], c[2]))
    return kind, v


def _remat_depth(var: int, info: RematInfo) -> int:
    """Peak simultaneous materialisation temps for recomputing ``var``.

    A leaf recomputation (or one with no operands) needs 1 temp; recomputing
    an operand that is itself rematerialised needs its subtree's temps on
    top.  This is a conservative *over*count (operand temps of one replay
    never coexist with operand temps of the next replay), so it never
    approves an intrinsically impossible point; it may only decline a point
    that a more exact model could accept.
    """
    memo = {}

    def depth(x: int) -> int:
        if x in memo:
            return memo[x]
        memo[x] = 1  # break hypothetical cycles (single-def SSA has none)
        extra = 0
        for u in info.operands.get(x, ()):
            if u in info.eligible:
                extra = max(extra, depth(u))
        memo[x] = 1 + extra
        return memo[x]

    return depth(var)


def _remat_intrinsic_feasible(var: int, program: Program, spilled_after: set[int],
                              info: RematInfo, k: int) -> bool:
    """Conservative intrinsic register-need check at every point that would
    recompute ``var``.

    A use point cannot be made cheaper by spilling *other* values: the
    values read by the consuming instruction must coexist.  We count the
    distinct simultaneous registers required when ``var`` (and its closure)
    is rematerialised, treating every other operand conservatively; if any
    point provably needs more than K registers, the remat action is
    declined and the iteration spills instead.  This never introduces a
    phantom temporary that the colorer cannot place.
    """
    r_set = set(_remat_closure(var, info)) | (spilled_after & set(info.eligible))

    def need_for(consuming_uses: tuple[int, ...]) -> int:
        need = 0
        for u in dict.fromkeys(consuming_uses):
            if u == var:
                need += _remat_depth(var, info)
            elif u in r_set:
                need += _remat_depth(u, info)
            elif u in spilled_after:
                need += 1  # reload temp
            else:
                need += 1  # live home register
        return need

    # instruction use points
    for bi, ii, _ops, live_here in info.consumption.get(var, ()):
        if ii is None:
            continue
        ins = program.blocks[bi].instrs[ii]
        survivors = set(live_here) - set(ins.uses)
        need = need_for(tuple(ins.uses))
        for v in survivors:
            if v in r_set:
                need += _remat_depth(v, info)
            elif v in spilled_after:
                need += 1
            else:
                need += 1
        if need > k:
            return False
    # phi-source points: the recomputed source temp coexists with the other
    # sources materialised on that edge and with values crossing the edge
    for (pred, succ), srcs in info.edge_sources.items():
        need = 0
        for s in srcs:
            if s == var:
                need += _remat_depth(var, info)
            elif s in r_set:
                need += _remat_depth(s, info)
            elif s in spilled_after:
                need += 1
            else:
                need += 1
        if need > k:
            return False
    return True


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

@dataclass
class Allocation:
    program: Program
    k: int
    assignment: dict[int, int]
    slots: dict[int, int]
    rematerialized: dict[int, float]
    allocated_program: "object"
    liveness: "object"
    iterations: int
    history: tuple[tuple[str, ...], ...]


def allocate(program: Program, k: int,
             remat: dict[int, float] | None = None) -> Allocation:
    """Allocate ``program`` to ``k`` physical registers.

    ``remat`` optionally marks pure instructions whose single result may be
    recomputed at its use points instead of being held in a register or
    spilled: a mapping ``{instruction uid: recomputation cost}``.  Marking an
    instruction is the caller's assertion that it is pure (no side effects,
    no mutable-state dependency); markers on non-pure instructions (multiple
    results, register clobbers, fixed registers, phi results, parameters,
    redefined variables) are rejected with :class:`ProgramFormatError`.
    A marker that is structurally valid but whose operands cannot be proven
    available at every use point is simply never selected.

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
    remat_info = analyze_remat(program, remat if remat is not None else {})

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
    use_weights = _use_weights(program)

    # Search over the two kinds of pressure relief (spill / rematerialise).
    # A rematerialisation decision is not reversible once chosen (the value
    # loses any persistent home), and an aggressive remat can drive the graph
    # into a dead end that plain spilling would have avoided; so we branch
    # with chronological backtracking: on a dead end we undo the most recent
    # remat decision and spill that variable instead.  The budget path still
    # raises BudgetExhaustedError (feasibility unknown), distinct from a
    # fully exhausted search (proven infeasible).
    #
    # A search state is (spilled, remat_set, banned_remat).  banned_remat
    # records remat choices that already led to a dead end *with exactly the
    # current spilled set*; it is reset when spilling opens new possibilities.
    history: list[tuple[str, ...]] = []
    stack: list[tuple[frozenset[int], frozenset[int], frozenset[int]]] = [
        (frozenset(), frozenset(), frozenset())
    ]
    seen: set[tuple[frozenset[int], frozenset[int]]] = set()
    color: dict[int, int] | None = None
    vp: VirtualProgram | None = None
    spilled: set[int] = set()
    remat_set: set[int] = set()
    steps = 0
    max_steps = (program.var_count() + 1) * (program.var_count() + 2)

    while stack:
        steps += 1
        if steps > max_steps:  # pragma: no cover
            raise RegAllocError("internal: spill/remat search did not converge")
        s0, r0, banned = stack.pop()
        if (s0, r0) in seen:
            continue
        seen.add((s0, r0))
        spilled, remat_set = set(s0), set(r0)
        history.append(
            tuple(["spill:" + ",".join(map(str, sorted(spilled))),
                   "remat:" + ",".join(map(str, sorted(remat_set)))])
        )
        vp = build_virtual_program(program, spilled, remat_set, remat_info)

        clash = _precolor_conflicts(vp)
        if clash is not None:
            a, b, rc = clash
            raise InfeasibleProgramError(
                f"two fixed-register requirements collide in register r{rc} at an "
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
                raise BudgetExhaustedError(
                    f"exact coloring search budget exhausted with spilled vars "
                    f"{sorted(spilled)} remat vars {sorted(remat_set)}; "
                    f"feasibility unknown"
                )
                color = None
            else:
                color = None

        if color is not None:
            if any(v not in color for v in vp.adj):
                raise RegAllocError("internal: incomplete coloring")  # pragma: no cover
            break

        # dead end: no real variables left to evacuate on this branch
        if not vp.real_nodes:
            continue  # try a sibling branch (backtrack a remat decision)

        action = _choose_relief_action(vp, program, weights, use_weights,
                                       spilled, remat_set, remat_info, k,
                                       banned=banned)
        if action is None:
            # preferred relief is unavailable here; backtrack if possible
            continue
        kind, victim = action
        if kind == "spill":
            new_s = frozenset(spilled | {victim})
            stack.append((new_s, frozenset(remat_set), frozenset()))
        else:
            closure = frozenset(_remat_closure(victim, remat_info))
            new_r = frozenset(remat_set | closure)
            # explore the rematerialisation branch first, then the spill
            # branch for the same variable (chronological backtracking)
            stack.append((frozenset(spilled | {victim}), frozenset(remat_set), frozenset()))
            stack.append((frozenset(spilled), new_r, banned))

    if color is None:
        raise InfeasibleProgramError(
            "all spill/rematerialisation relief branches exhausted: proven no "
            "allocation exists"
        )

    if vp is None:  # pragma: no cover
        raise RegAllocError("internal: spill/remat search failed without a state")

    assignment = {v: color[v] for v in vp.real_nodes if v in color}
    # slots are indexed by variable id, as required by the contract
    slots = {v: v for v in sorted(spilled)}
    rematerialized = {v: remat_info.costs[remat_info.eligible[v]]
                      for v in sorted(remat_set)}

    # imported here to avoid a module cycle at import time
    from .lower import lower_program

    allocated_program = lower_program(program, vp, color, remat_info)

    allocation = Allocation(
        program=program,
        k=k,
        assignment=dict(sorted(assignment.items())),
        slots=dict(sorted(slots.items())),
        rematerialized=dict(sorted(rematerialized.items())),
        allocated_program=allocated_program,
        liveness=liveness_with_phis(program),
        iterations=steps,
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
        if v in allocation.slots or v in allocation.rematerialized:
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
            pairs = {(t[1], t[2]) for t in plan.pre + plan.post if t[0] == "copy"}
            for v, r in ins.fixed.items():
                if v in allocation.slots:
                    continue
                if allocation.assignment.get(v) == r:
                    continue
                if not any(src == v or dst == v for dst, src in pairs):
                    raise RegAllocError(  # pragma: no cover
                        f"internal: fixed requirement v{v}@r{r} not materialized"
                    )
