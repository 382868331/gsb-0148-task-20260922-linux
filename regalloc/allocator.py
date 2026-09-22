"""Core allocator: phi lowering, interference graph, Chaitin coloring,
iterative spilling and concrete code generation.

Pipeline
--------
1. Validate the program (:mod:`regalloc.validate`).
2. Edge-sensitive liveness on the ORIGINAL program (:mod:`regalloc.liveness`)
   -- returned to the caller so it can be checked independently.
3. Iteratively (the classic Chaitin spill loop; the interference graph is
   rebuilt from scratch every round):
   a. Lower phis to per-edge *parallel copies*. Critical edges are split with
      a dedicated block; copies on non-critical edges are placed at the
      predecessor's tail (or successor's head).
   b. Replace occurrences of spilled variables with short-lived
      load/use/store temporaries.
   c. Run plain liveness on the expanded phi-free CFG and collect exact
      coexistence (interference) sets at every instruction point; fixed
      registers and call-clobber sets become forced/forbidden colors.
   d. Deterministic simplify/select coloring. Failure spills one more
      original variable (smallest degree/occurrence-weight ratio, name
      tie-break) and rebuilds.
4. Emit concrete :class:`MInstr` code. Parallel copies are sequentialized;
   residual permutation cycles (plain phi lowering never produces any, but
   the mechanism is implemented regardless) are broken with one deterministic
   scratch stack slot.

Interference, precisely
-----------------------
Let L/A be live-before/live-after and W the written variables.

At an instruction point:

* the values in ``L`` are all physically present when the inputs are read,
  so they form a clique (operands that die at the instruction included);
* the results ``W`` are born together, so they form a clique;
* a result must not overwrite a value that survives the point, so every
  member of W clashes with every member of ``L & A``.

A result MAY reuse the register of an operand that dies there (``L - A``),
so no edge is added between W and dying operands. The same rule at a
parallel-copy point means a dying copy source ``s_i`` is not connected even
to a *different* destination ``d_j``. That is safe because the
sequentializer never emits ``d_j <- s_j`` while register ``color(s_i)`` is
still a pending source (the move that reads ``s_i`` is always drained
first); a pure swap/cycle is broken via the scratch slot. It only forgoes a
coalescing opportunity, never correctness. When the copy source survives
(``s_i`` in ``L & A``) the normal W-vs-survivor edge exists.

Clobbers: only a value that survives an instruction (``L & A``) is banned
from the clobbered registers. A dying operand is gone anyway, and a result
may well be delivered in a clobbered (caller-saved) register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .liveness import compute_liveness
from .model import (
    AllocatedProgram,
    Allocation,
    AllocationBudgetExhausted,
    Block,
    Instruction,
    MBlock,
    MInstr,
    Program,
    Reg,
    Slot,
    UnsatisfiableProgram,
)
from .validate import validate_program

# Prefixes reserved for allocator-synthetic names (rejected in user input).
_TMP_PREFIX = ".."
_SPLIT_PREFIX = _TMP_PREFIX + "split."

#: Search budget for the spill/recolor loop. Every successful round spills a
#: distinct original variable, so the loop ends within ``n_vars + 1`` rounds;
#: the cap only guards defensive paths. Hitting it raises
#: AllocationBudgetExhausted, which is distinct from UnsatisfiableProgram.
MAX_ITERATIONS = 10_000


# --------------------------------------------------------------------------- #
# Expanded (phi-free, spill-lowered) IR
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EInstr:
    """An instruction of the expanded CFG.

    kinds:
      * ``"orig"``  : a user instruction (occurrence index in ``origin``), or
        the synthetic ``"param"`` entry materialization (origin None).
      * ``"par"``   : a parallel copy; ``reads[i] -> writes[i]`` are aligned
        pairs executed simultaneously.
      * ``"load"``  : fresh temporary ``writes[0]`` gets the value of the
        spilled variable ``slot_var`` from its stack slot.
      * ``"store"`` : spilled variable ``slot_var`` gets ``reads[0]``'s value.
    """

    kind: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    origin: Optional[int] = None
    opcode: str = ""
    fixed_use: dict[str, int] = field(default_factory=dict)
    fixed_def: dict[str, int] = field(default_factory=dict)
    clobbers: tuple[int, ...] = ()
    slot_var: Optional[str] = None


@dataclass
class EBlock:
    id: str
    instrs: list[EInstr] = field(default_factory=list)
    succs: list[str] = field(default_factory=list)


def _expanded_liveness(eblocks: list[EBlock]):
    """Backward liveness on the phi-free expanded CFG.

    Per-point live-before/live-after come from a reverse walk off each
    block's fixpoint live-out, so surviving definitions are part of
    live-after. Returns ``(live_in, live_out, before, after)`` keyed by block
    id and ``(block id, instr index)``. Parameters need no boundary forcing:
    they are born at explicit ``"param"`` glue at the head of the entry
    block, so a dead parameter never becomes live.
    """
    in_sets: dict[str, frozenset[str]] = {b.id: frozenset() for b in eblocks}
    out_sets: dict[str, frozenset[str]] = {b.id: frozenset() for b in eblocks}

    changed = True
    while changed:
        changed = False
        for b in reversed(eblocks):
            cur: set[str] = set()
            for s in b.succs:
                cur |= in_sets[s]
            new_out = frozenset(cur)
            for ei in reversed(b.instrs):
                cur.difference_update(ei.writes)
                cur.update(ei.reads)
            new_in = frozenset(cur)
            if new_in != in_sets[b.id] or new_out != out_sets[b.id]:
                in_sets[b.id] = new_in
                out_sets[b.id] = new_out
                changed = True

    before: dict[tuple[str, int], frozenset[str]] = {}
    after: dict[tuple[str, int], frozenset[str]] = {}
    for b in eblocks:
        cur = set(out_sets[b.id])
        for i in range(len(b.instrs) - 1, -1, -1):
            ei = b.instrs[i]
            after[(b.id, i)] = frozenset(cur)
            cur = set(cur)
            cur.difference_update(ei.writes)
            cur.update(ei.reads)
            before[(b.id, i)] = frozenset(cur)
    return in_sets, out_sets, before, after


# --------------------------------------------------------------------------- #
# Expansion: phi lowering + spill occurrence lowering
# --------------------------------------------------------------------------- #


class _Expander:
    def __init__(self, program: Program, spilled: frozenset[str]):
        self.program = program
        self.spilled = spilled
        self.counter = 0
        self.preds: dict[str, list[str]] = {b.id: [] for b in program.blocks}
        for b in program.blocks:
            for s in b.succs:
                self.preds[s].append(b.id)

    def _fresh(self, var: str) -> str:
        self.counter += 1
        return f"{_TMP_PREFIX}{var}#{self.counter}"

    def expand(self) -> list[EBlock]:
        # ---- decide edge-copy placement -----------------------------------
        edge_ops: dict[tuple[str, str], list[EInstr]] = {}
        placement: dict[tuple[str, str], str] = {}
        block_by_id = {b.id: b for b in self.program.blocks}
        for b in self.program.blocks:
            if not b.phis:
                continue
            for p in self.preds[b.id]:
                edge_ops[(p, b.id)] = self._edge_phi_ops(p, b)
                pb = block_by_id[p]
                if len(pb.succs) == 1:
                    placement[(p, b.id)] = "tail"
                elif len(self.preds[b.id]) == 1:
                    placement[(p, b.id)] = "head"
                else:
                    placement[(p, b.id)] = "split"

        head_ops: dict[str, list[EInstr]] = {}
        tail_ops: dict[str, list[EInstr]] = {}
        split_edges: list[tuple[str, str]] = []
        for edge, where in placement.items():
            if where == "head":
                head_ops.setdefault(edge[1], []).extend(edge_ops[edge])
            elif where == "tail":
                tail_ops.setdefault(edge[0], []).extend(edge_ops[edge])
            else:
                split_edges.append(edge)

        # ---- original blocks, in original order ---------------------------
        eblocks: list[EBlock] = []
        for b in self.program.blocks:
            eb = EBlock(id=b.id)
            if b.id == self.program.entry:
                # Materialize every parameter exactly once, in declared order:
                # unspilled params are born into a register; spilled params
                # into a short-lived temp immediately stored to their slot.
                for p in self.program.params:
                    if p in self.spilled:
                        t = self._fresh(p)
                        eb.instrs.append(EInstr("orig", (), (t,), opcode="param"))
                        eb.instrs.append(EInstr("store", (t,), (), slot_var=p))
                    else:
                        eb.instrs.append(EInstr("orig", (), (p,), opcode="param"))
            eb.instrs.extend(head_ops.get(b.id, ()))
            for j, ins in enumerate(b.instrs):
                eb.instrs.extend(self._lower_instr(j, ins))
            eb.instrs.extend(tail_ops.get(b.id, ()))
            eb.succs = list(b.succs)
            eblocks.append(eb)

        # ---- critical-edge split blocks (deterministic edge order) --------
        split_id: dict[tuple[str, str], str] = {}
        for p, q in split_edges:
            sid = f"{_SPLIT_PREFIX}{p}->{q}"
            split_id[(p, q)] = sid
            sb = EBlock(id=sid)
            sb.instrs.extend(edge_ops[(p, q)])
            sb.succs = [q]
            eblocks.append(sb)
        for eb in eblocks:
            eb.succs = [split_id[(eb.id, s)] if (eb.id, s) in split_id else s for s in eb.succs]
        return eblocks

    def _edge_phi_ops(self, pred: str, b: Block) -> list[EInstr]:
        """Lowered sequence on edge (pred -> b):

        loads for spilled sources ; one simultaneous parallel copy ; stores
        for spilled destinations.
        """
        out: list[EInstr] = []
        pairs: list[tuple[str, str]] = []
        dst_spilled: list[bool] = []
        for phi in b.phis:
            src = phi.srcs[pred]
            if src in self.spilled:
                t = self._fresh(src)
                out.append(EInstr("load", (), (t,), slot_var=src))
                src = t
            if phi.dst in self.spilled:
                t2 = self._fresh(phi.dst)
                pairs.append((src, t2))
                dst_spilled.append(True)
            else:
                pairs.append((src, phi.dst))
                dst_spilled.append(False)

        if pairs:
            fixed_def: dict[str, int] = {}
            for phi, (_, d) in zip(b.phis, pairs):
                if phi.fixed_def is not None:
                    fixed_def[d] = phi.fixed_def
            out.append(
                EInstr(
                    "par",
                    tuple(s for s, _ in pairs),
                    tuple(d for _, d in pairs),
                    fixed_def=fixed_def,
                )
            )
        for phi, (_, d), was_spilled in zip(b.phis, pairs, dst_spilled):
            if was_spilled:
                out.append(EInstr("store", (d,), (), slot_var=phi.dst))
        return out

    def _lower_instr(self, index: int, ins: Instruction) -> list[EInstr]:
        out: list[EInstr] = []

        reads: list[str] = []
        fixed_use: dict[str, int] = {}
        for u in ins.uses:
            if u in self.spilled:
                t = self._fresh(u)
                if u in ins.fixed_use:
                    fixed_use[t] = ins.fixed_use[u]
                out.append(
                    EInstr(
                        "load",
                        (),
                        (t,),
                        fixed_def=({t: ins.fixed_use[u]} if u in ins.fixed_use else {}),
                        slot_var=u,
                    )
                )
                reads.append(t)
            else:
                reads.append(u)
                if u in ins.fixed_use:
                    fixed_use[u] = ins.fixed_use[u]

        writes: list[str] = []
        fixed_def: dict[str, int] = {}
        store_terms: list[tuple[str, str]] = []
        for d in ins.defs:
            if d in self.spilled:
                t = self._fresh(d)
                writes.append(t)
                store_terms.append((d, t))
                if d in ins.fixed_def:
                    fixed_def[t] = ins.fixed_def[d]
            else:
                writes.append(d)
                if d in ins.fixed_def:
                    fixed_def[d] = ins.fixed_def[d]

        out.append(
            EInstr(
                "orig",
                tuple(reads),
                tuple(writes),
                origin=index,
                opcode=ins.opcode,
                fixed_use=fixed_use,
                fixed_def=fixed_def,
                clobbers=ins.clobbers,
            )
        )
        for d, t in store_terms:
            out.append(EInstr("store", (t,), (), slot_var=d))
        return out


# --------------------------------------------------------------------------- #
# Interference graph
# --------------------------------------------------------------------------- #


def _build_graph(eblocks, before, after):
    """Build interference edges and clobber bans.

    Returns ``(edges, ban, nodes)``. See module docstring for the exact
    coexistence rules.
    """
    pair_set: set[tuple[str, str]] = set()
    ban: dict[str, set[int]] = {}
    nodes: set[str] = set()

    def clique(vals) -> None:
        vals = list(vals)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, c = vals[i], vals[j]
                pair_set.add((a, c) if a < c else (c, a))

    for b in eblocks:
        for i, ei in enumerate(b.instrs):
            L = set(before[(b.id, i)])
            A = set(after[(b.id, i)])
            nodes.update(L, A, ei.reads, ei.writes)
            W = set(ei.writes)
            through = L & A  # values present both before and after the point

            clique(L)                 # operands + live-through at read time
            clique(W)                 # results born together
            for w in W:               # results vs surviving values
                for t in through:
                    pair_set.add((w, t) if w < t else (t, w))
            # No edges from W to dying operands (L - A): a result may reuse
            # the register of an operand that dies; for a parallel copy this
            # is exactly the (s_i -> d_i) coalescing exemption.

            if ei.kind == "orig" and ei.clobbers:
                for v in through:
                    ban.setdefault(v, set()).update(ei.clobbers)

    edges: dict[str, set[str]] = {v: set() for v in nodes}
    for a, c in pair_set:
        edges[a].add(c)
        edges[c].add(a)
    for v in nodes:
        ban.setdefault(v, set())
    return edges, ban, nodes


# --------------------------------------------------------------------------- #
# Coloring
# --------------------------------------------------------------------------- #


def _colorize(nodes, edges, fixed, ban, K, spillable, weight):
    """Deterministic Chaitin simplify/select.

    Simplify always removes a lowest-degree (< K) node; when none remains the
    spillable original with the smallest degree/weight ratio is optimistically
    pushed. Select pops in reverse, honoring fixed colors and clobber bans.
    Returns ``(colors, uncolored)``.
    """
    remaining = set(nodes)
    stack: list[str] = []

    def degree(v: str) -> int:
        return len(edges.get(v, ()) & remaining)

    while remaining:
        low = [v for v in remaining if degree(v) < K]
        if low:
            chosen = min(low)
        else:
            cands = [v for v in remaining if spillable(v)]
            if cands:
                chosen = min(cands, key=lambda v: (degree(v) / weight(v), v))
            else:
                chosen = min(remaining)
        remaining.remove(chosen)
        stack.append(chosen)

    colors: dict[str, int] = {}
    uncolored: list[str] = []
    while stack:
        v = stack.pop()
        used = {colors[u] for u in edges.get(v, ()) if u in colors}
        if v in fixed:
            r = fixed[v]
            if r in used or r in ban.get(v, set()):
                uncolored.append(v)
                continue
            colors[v] = r
        else:
            avail = set(range(K)) - used - ban.get(v, set())
            if avail:
                colors[v] = min(avail)
            else:
                uncolored.append(v)
    return colors, uncolored


# --------------------------------------------------------------------------- #
# Parallel-copy sequentialization
# --------------------------------------------------------------------------- #


def _sequentialize_par(ei: EInstr, colors: dict[str, int], scratch_slot: int):
    """Turn a simultaneous copy group into ordered concrete moves.

    Destination registers are distinct; source registers may repeat. The
    safe rule: emit ``d <- s`` once no other still-pending move reads the old
    value of register ``d`` (i.e. ``d`` is no longer among the pending
    sources). When no such move exists, the remaining mapping is one or more
    permutation cycles: save one cycle value to a scratch stack slot, drain
    the newly-ready chain, and -- only after the whole group is drained --
    reload it. One scratch slot suffices because cycles are broken one at a
    time.
    """
    out: list[MInstr] = []
    pending: dict[int, int] = {}
    for s, d in zip(ei.reads, ei.writes):
        rs, rd = colors[s], colors[d]
        if rs != rd:
            pending.setdefault(rd, rs)

    def mov(sc: int, dc: int) -> None:
        out.append(MInstr("mov", (Reg(sc),), (Reg(dc),), None))

    used_scratch = False
    while pending:
        ready = [d for d in pending if d not in set(pending.values())]
        if ready:
            for d in sorted(ready):
                mov(pending[d], d)
                del pending[d]
            continue
        # Pure cycle remaining: break it at the smallest destination.
        d0 = min(pending)
        s0 = pending[d0]
        used_scratch = True
        out.append(MInstr("store", (Reg(s0),), (Slot(scratch_slot),), None))
        del pending[d0]
        # Drain the chain made ready by removing d0 before reloading, so the
        # reload never overwrites a value a pending move still has to read.
        while True:
            ready2 = [d for d in pending if d not in set(pending.values())]
            if not ready2:
                break
            for d in sorted(ready2):
                mov(pending[d], d)
                del pending[d]
        out.append(MInstr("load", (Slot(scratch_slot),), (Reg(d0),), None))
        # If another disjoint cycle remains, the outer loop handles it; the
        # reload above is safe because that cycle never touches d0.
    return out, used_scratch


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def allocate(program: Program, K) -> Allocation:
    """Allocate ``K`` physical registers (2 <= K <= 8) for ``program``.

    Raises :class:`InvalidArgumentError` / :class:`InvalidProgramError` for
    bad input, :class:`UnsatisfiableProgram` when the fixed/clobber contract
    is contradictory in a way spilling cannot repair, and
    :class:`AllocationBudgetExhausted` only if the bounded spill search gives
    up (never a claim of unsatisfiability).
    """
    program = validate_program(program, K)
    live = compute_liveness(program)

    # Occurrence weights for the deterministic spill heuristic.
    weight: dict[str, float] = {v: 0.0 for v in program.params}
    for b in program.blocks:
        for phi in b.phis:
            weight[phi.dst] = weight.get(phi.dst, 0.0) + 1.0
            for src in phi.srcs.values():
                weight[src] = weight.get(src, 0.0) + 1.0
        for ins in b.instrs:
            for u in ins.uses:
                weight[u] = weight.get(u, 0.0) + 1.0
            for d in ins.defs:
                weight[d] = weight.get(d, 0.0) + 1.0
    for v in weight:
        if weight[v] <= 0.0:
            weight[v] = 1.0

    def is_spillable(v: str) -> bool:
        # Synthetic occurrence temps die within one instruction and are never
        # spill candidates; original variables (parameters included, which
        # are re-materialized at entry) are.
        return not v.startswith(_TMP_PREFIX)

    spilled: set[str] = set()
    iterations = 0
    colors: dict[str, int] = {}

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            raise AllocationBudgetExhausted(
                f"spill/recolor search exceeded {MAX_ITERATIONS} iterations "
                "without a coloring (this is not a claim of unsatisfiability)"
            )

        eblocks = _Expander(program, frozenset(spilled)).expand()
        in_sets, _out_sets, before, after = _expanded_liveness(eblocks)
        edges, ban, nodes = _build_graph(eblocks, before, after)

        fixed: dict[str, int] = {}
        for b in eblocks:
            for ei in b.instrs:
                for v, r in ei.fixed_use.items():
                    fixed[v] = r
                for v, r in ei.fixed_def.items():
                    fixed[v] = r
        for v in fixed:
            nodes.add(v)
            edges.setdefault(v, set())
            ban.setdefault(v, set())

        # Provable impossibilities (structural, independent of search luck):
        # clashing values forced into one register, or a forced value that
        # must survive across a clobber of its register.
        for v, r in fixed.items():
            if r in ban.get(v, set()):
                raise UnsatisfiableProgram(
                    f"value {v!r} is forced into r{r} but must survive an "
                    f"instruction that clobbers r{r}"
                )
        fixed_by_color: dict[int, list[str]] = {}
        for v, r in fixed.items():
            fixed_by_color.setdefault(r, []).append(v)
        for r, vs in fixed_by_color.items():
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a, c = vs[i], vs[j]
                    if c in edges.get(a, set()):
                        raise UnsatisfiableProgram(
                            f"simultaneously live values {a!r} and {c!r} are "
                            f"both forced into r{r}"
                        )

        colors, uncolored = _colorize(
            nodes,
            edges,
            fixed,
            ban,
            K,
            spillable=is_spillable,
            weight=lambda v: weight.get(v, 1.0),
        )
        if not uncolored:
            break

        uncolored_set = set(uncolored)
        victims = [v for v in uncolored_set if is_spillable(v)]
        if not victims:
            # No failed node is itself spillable (typical case: the failed
            # nodes are single-point reload temps). Spilling an original
            # live-through NEIGHBOR of a failed node can still free a color;
            # only when no such neighbor exists is the point genuinely
            # infeasible (e.g. more than K operands meeting at one use).
            victims = sorted(
                v
                for v in nodes
                if is_spillable(v)
                and any(u in uncolored_set for u in edges.get(v, ()))
            )
        if not victims:
            raise UnsatisfiableProgram(
                "register pressure at a single program point exceeds K in a "
                "way spilling cannot repair: every instruction operand must "
                "sit in a register at its use (a reload temp occupies a "
                "register at that exact point), and every value blocking a "
                "free color is itself such a point-local operand. "
                f"Uncolorable values: {sorted(uncolored)}"
            )
        victim = min(
            victims,
            key=lambda v: (len(edges.get(v, ())) / weight.get(v, 1.0), v),
        )
        spilled.add(victim)

    # ---- stack slots: spilled originals ordered by variable id ------------
    spilled_sorted = sorted(spilled)
    slot_of = {v: i for i, v in enumerate(spilled_sorted)}
    scratch_slot = len(spilled_sorted)

    # ---- concrete code generation (re-expand from the successful state) ----
    eblocks = _Expander(program, frozenset(spilled)).expand()
    mblocks: list[MBlock] = []
    used_scratch = False
    for b in eblocks:
        mins: list[MInstr] = []
        for ei in b.instrs:
            if ei.kind == "orig":
                mins.append(
                    MInstr(
                        ei.opcode,
                        tuple(Reg(colors[r]) for r in ei.reads),
                        tuple(Reg(colors[w]) for w in ei.writes),
                        ei.origin,
                        ei.clobbers,
                    )
                )
            elif ei.kind == "load":
                mins.append(
                    MInstr(
                        "load",
                        (Slot(slot_of[ei.slot_var]),),
                        (Reg(colors[ei.writes[0]]),),
                        None,
                    )
                )
            elif ei.kind == "store":
                mins.append(
                    MInstr(
                        "store",
                        (Reg(colors[ei.reads[0]]),),
                        (Slot(slot_of[ei.slot_var]),),
                        None,
                    )
                )
            else:  # par
                seq, scratched = _sequentialize_par(ei, colors, scratch_slot)
                used_scratch |= scratched
                mins.extend(seq)
        mblocks.append(MBlock(b.id, tuple(mins), tuple(b.succs)))

    assignment = {v: colors.get(v) for v in weight}
    # Variables that are never live (dead code / unused parameters) need no
    # documented register even though their defining instruction still
    # physically writes somewhere.
    ever_live: set[str] = set()
    for t in live.live_in.values():
        ever_live.update(t)
    for t in live.live_out.values():
        ever_live.update(t)
    for before, after in live.points.values():
        ever_live.update(before)
        ever_live.update(after)
    for t in live.edge_live.values():
        ever_live.update(t)
    for v in assignment:
        if v not in ever_live:
            assignment[v] = None
    for v in spilled:
        assignment[v] = None

    return Allocation(
        program=AllocatedProgram(program.entry, tuple(mblocks)),
        assignment=dict(sorted(assignment.items())),
        stack_slots={v: slot_of[v] for v in spilled_sorted},
        live=live,
        K=K,
        iterations=iterations,
        scratch_slots=(scratch_slot,) if used_scratch else (),
    )
