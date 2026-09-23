"""Liveness analysis.

Two variants are provided:

* :func:`liveness_plain` for phi-free CFGs (the IR after phi lowering);
* :func:`liveness_with_phis` for the original CFG where phi inputs are bound
  to predecessor *edges*.  Phi sources are live on their edge only and never
  on the other incoming edges — this is the edge-sensitive part.

Both produce the same independently checkable report shape: per-block
live-in/live-out, per-edge live sets and per-instruction live-before/live-after
sets.  Fixed-point iteration visits blocks in ascending index order, so
results are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Block, Program


@dataclass(frozen=True)
class LivenessReport:
    """Checkable liveness result for one program.

    All members map by *block index* (and, for ``instr_before``, by
    instruction position inside the block).
    """

    live_in: tuple[frozenset[int], ...]
    live_out: tuple[frozenset[int], ...]
    edge_live: dict[tuple[int, int], frozenset[int]]
    instr_before: tuple[tuple[frozenset[int], ...], ...]
    instr_after: tuple[tuple[frozenset[int], ...], ...]

    def as_jsonable(self) -> dict:
        return {
            "live_in": [sorted(s) for s in self.live_in],
            "live_out": [sorted(s) for s in self.live_out],
            "edge_live": {f"{p}->{s}": sorted(v) for (p, s), v in sorted(self.edge_live.items())},
            "instr_before": [[sorted(s) for s in row] for row in self.instr_before],
            "instr_after": [[sorted(s) for s in row] for row in self.instr_after],
        }


def predecessors(blocks: tuple[Block, ...]) -> list[list[int]]:
    preds: list[list[int]] = [[] for _ in blocks]
    for i, b in enumerate(blocks):
        for s in b.succs:
            preds[s].append(i)
    for lst in preds:
        lst.sort()
    return preds


def _fixpoint(blocks, preds, transfer_out, up_exp, killed):
    """Backward liveness fixpoint over block live sets.

    ``transfer_out(i, li_successor_values)`` computes live-out for block i;
    ``up_exp[i]`` / ``killed[i]`` are the block body transfer functions.
    """
    n = len(blocks)
    live_in: list[set[int]] = [set() for _ in range(n)]
    live_out: list[set[int]] = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            new_out = transfer_out(i, live_in)
            new_in = up_exp[i] | (new_out - killed[i])
            if new_out != live_out[i] or new_in != live_in[i]:
                live_out[i] = set(new_out)
                live_in[i] = set(new_in)
                changed = True
    return live_in, live_out


def liveness_plain(program: Program) -> LivenessReport:
    """Liveness for a phi-free program (copy/load/store/generic ops)."""
    blocks = program.blocks
    n = len(blocks)
    preds = predecessors(blocks)

    up_exp: list[set[int]] = []
    killed: list[set[int]] = []
    for b in blocks:
        killed_b: set[int] = set()
        up_b: set[int] = set()
        for ins in b.instrs:
            up_b.update(v for v in ins.uses if v not in killed_b)
            killed_b.update(ins.defs)
        up_exp.append(up_b)
        killed.append(killed_b)

    def out_of(i, live_in):
        out: set[int] = set()
        for s in blocks[i].succs:
            out |= live_in[s]
        return out

    live_in, live_out = _fixpoint(blocks, preds, out_of, up_exp, killed)

    edge_live: dict[tuple[int, int], frozenset[int]] = {}
    for i, b in enumerate(blocks):
        for s in b.succs:
            edge_live[(i, s)] = frozenset(live_in[s])

    instr_before: list[tuple[frozenset[int], ...]] = []
    instr_after: list[tuple[frozenset[int], ...]] = []
    for i, b in enumerate(blocks):
        live = set(live_out[i])
        befores: list[frozenset[int]] = []
        afters: list[frozenset[int]] = []
        for ins in reversed(b.instrs):
            afters.append(frozenset(live))
            live = (live - set(ins.defs)) | set(ins.uses)
            befores.append(frozenset(live))
        instr_before.append(tuple(reversed(befores)))
        instr_after.append(tuple(reversed(afters)))

    return LivenessReport(
        live_in=tuple(frozenset(s) for s in live_in),
        live_out=tuple(frozenset(s) for s in live_out),
        edge_live=edge_live,
        instr_before=tuple(instr_before),
        instr_after=tuple(instr_after),
    )


def liveness_with_phis(program: Program) -> LivenessReport:
    """Edge-sensitive liveness for programs containing phi nodes.

    Equations (Cooper/Torczon style)::

        out(B) = union over succ S of (in(S) - phiDefs(S)) ∪ phiSrcs(B -> S)
        in(B)  = upExp(B) ∪ (out(B) - (bodyDefs(B) ∪ phiDefs(B)))

    where ``phiSrcs(B -> S)`` contains exactly the source variables of S's
    phi nodes on the edge coming from B.  A phi source is therefore live on
    one predecessor edge only.
    """
    blocks = program.blocks
    n = len(blocks)
    preds = predecessors(blocks)

    phi_defs: list[set[int]] = []
    # sources_by_edge[(s, p)] = list of source vars on edge p -> s
    sources_on_edge: dict[tuple[int, int], list[int]] = {}
    for i, b in enumerate(blocks):
        d: set[int] = set()
        for phi in b.phis:
            d.add(phi.def_var)
            for p, src in phi.sources:
                sources_on_edge.setdefault((i, p), []).append(src)
        phi_defs.append(d)

    up_exp: list[set[int]] = []
    killed: list[set[int]] = []
    for i, b in enumerate(blocks):
        killed_b = set(phi_defs[i])
        up_b: set[int] = set()
        for ins in b.instrs:
            up_b.update(v for v in ins.uses if v not in killed_b)
            killed_b.update(ins.defs)
        up_exp.append(up_b)
        killed.append(killed_b)

    def out_of(i, live_in):
        out: set[int] = set()
        for s in blocks[i].succs:
            edge = (live_in[s] - phi_defs[s]) | set(sources_on_edge.get((s, i), ()))
            out |= edge
        return out

    live_in, live_out = _fixpoint(blocks, preds, out_of, up_exp, killed)

    edge_live: dict[tuple[int, int], frozenset[int]] = {}
    for i, b in enumerate(blocks):
        for s in b.succs:
            edge_live[(i, s)] = frozenset(
                (live_in[s] - phi_defs[s]) | set(sources_on_edge.get((s, i), ()))
            )

    # Per-instruction sets: scan each block backward from the join of its
    # outgoing edge live sets; phi defs are defined before the body.
    instr_before: list[tuple[frozenset[int], ...]] = []
    instr_after: list[tuple[frozenset[int], ...]] = []
    for i, b in enumerate(blocks):
        live: set[int] = set()
        for s in b.succs:
            live |= edge_live[(i, s)]
        befores: list[frozenset[int]] = []
        afters: list[frozenset[int]] = []
        for ins in reversed(b.instrs):
            afters.append(frozenset(live))
            live = (live - set(ins.defs)) | set(ins.uses)
            befores.append(frozenset(live))
        # phi definitions exist at block entry and must not read as live-in
        entry = frozenset(live - phi_defs[i])
        assert entry == frozenset(live_in[i]), (i, entry, live_in[i])
        instr_before.append(tuple(reversed(befores)))
        instr_after.append(tuple(reversed(afters)))

    return LivenessReport(
        live_in=tuple(frozenset(s) for s in live_in),
        live_out=tuple(frozenset(s) for s in live_out),
        edge_live=edge_live,
        instr_before=tuple(instr_before),
        instr_after=tuple(instr_after),
    )


def predecessors_v2(blocks: tuple[Block, ...]) -> list[list[int]]:
    """Predecessors over normal AND exceptional edges (v2)."""
    preds: list[list[int]] = [[] for _ in blocks]
    for i, b in enumerate(blocks):
        for s in b.succs:
            preds[s].append(i)
        for s in b.exc_succs:
            preds[s].append(i)
    for lst in preds:
        lst.sort()
    return preds


def throw_index(block: Block) -> int | None:
    """Index of the block's may_throw instruction, or None."""
    for i, ins in enumerate(block.instrs):
        if ins.may_throw:
            return i
    return None


def liveness_v2(program: Program) -> LivenessReport:
    """Edge-sensitive liveness for v2 programs (exceptional control flow).

    Normal edges behave exactly as in :func:`liveness_with_phis`.  An
    exceptional edge ``p -> q`` leaves block ``p`` at its may_throw
    instruction ``t`` (which need not be the last instruction):

    * the exceptional edge live set is
      ``(in(q) - phiDefs(q)) ∪ phiSrcs(p -> q)``, taken at the program
      point immediately after instruction ``t``;
    * values needed on the exceptional edge become live before ``t`` and
      propagate to ``p``'s live-in (a value read by a handler must survive
      the potentially-throwing call);
    * ``instr_after[t]`` is the union of the normal and the exceptional
      continuations (a conservative may-set; documented, sound);
    * instructions after ``t`` in the same block are normal-path-only.

    The fixpoint iterates blocks in ascending index order; results are
    deterministic.
    """
    blocks = program.blocks
    n = len(blocks)

    phi_defs: list[set[int]] = []
    sources_on_edge: dict[tuple[int, int], list[int]] = {}
    for i, b in enumerate(blocks):
        d: set[int] = set()
        for phi in b.phis:
            d.add(phi.def_var)
            for p, src in phi.sources:
                sources_on_edge.setdefault((i, p), []).append(src)
        phi_defs.append(d)

    def edge_set(pred: int, succ: int) -> set[int]:
        return (live_in[succ] - phi_defs[succ]) | set(
            sources_on_edge.get((succ, pred), ())
        )

    throw_at = [throw_index(b) for b in blocks]

    live_in: list[set[int]] = [set() for _ in range(n)]
    live_out: list[set[int]] = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            b = blocks[i]
            new_out: set[int] = set()
            for s in b.succs:
                new_out |= edge_set(i, s)
            # backward scan; inject exceptional requirements at the throw
            live = set(new_out)
            t = throw_at[i]
            if t is None:
                for k in range(len(b.instrs) - 1, -1, -1):
                    ins = b.instrs[k]
                    live = (live - set(ins.defs)) | set(ins.uses)
            else:
                for k in range(len(b.instrs) - 1, t, -1):
                    ins = b.instrs[k]
                    live = (live - set(ins.defs)) | set(ins.uses)
                for q in b.exc_succs:
                    live |= edge_set(i, q)
                for k in range(t, -1, -1):
                    ins = b.instrs[k]
                    live = (live - set(ins.defs)) | set(ins.uses)
            new_in = live - phi_defs[i]
            if new_out != live_out[i] or new_in != live_in[i]:
                live_out[i] = set(new_out)
                live_in[i] = set(new_in)
                changed = True

    edge_live: dict[tuple[int, int], frozenset[int]] = {}
    for i, b in enumerate(blocks):
        for s in b.succs:
            edge_live[(i, s)] = frozenset(edge_set(i, s))
        for q in b.exc_succs:
            edge_live[(i, q)] = frozenset(edge_set(i, q))

    # Per-instruction sets, recomputed from the same equations.
    instr_before: list[tuple[frozenset[int], ...]] = []
    instr_after: list[tuple[frozenset[int], ...]] = []
    for i, b in enumerate(blocks):
        live: set[int] = set()
        for s in b.succs:
            live |= edge_live[(i, s)]
        t = throw_at[i]
        befores: list[frozenset[int]] = [frozenset()] * len(b.instrs)
        afters: list[frozenset[int]] = [frozenset()] * len(b.instrs)
        if t is None:
            for k in range(len(b.instrs) - 1, -1, -1):
                ins = b.instrs[k]
                afters[k] = frozenset(live)
                live = (live - set(ins.defs)) | set(ins.uses)
                befores[k] = frozenset(live)
        else:
            for k in range(len(b.instrs) - 1, t, -1):
                ins = b.instrs[k]
                afters[k] = frozenset(live)
                live = (live - set(ins.defs)) | set(ins.uses)
                befores[k] = frozenset(live)
            for q in b.exc_succs:
                live |= set(edge_live[(i, q)])
            for k in range(t, -1, -1):
                ins = b.instrs[k]
                afters[k] = frozenset(live)
                live = (live - set(ins.defs)) | set(ins.uses)
                befores[k] = frozenset(live)
        entry = frozenset(live - phi_defs[i])
        assert entry == frozenset(live_in[i]), (i, entry, live_in[i])
        instr_before.append(tuple(befores))
        instr_after.append(tuple(afters))

    return LivenessReport(
        live_in=tuple(frozenset(s) for s in live_in),
        live_out=tuple(frozenset(s) for s in live_out),
        edge_live=edge_live,
        instr_before=tuple(instr_before),
        instr_after=tuple(instr_after),
    )
