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
