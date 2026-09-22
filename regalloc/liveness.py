"""Edge-sensitive liveness analysis for the ORIGINAL three-address IR.

Fixed, independently checkable semantics:

* A phi destination is *defined* at its block's entry. It is live inside the
  block from the entry on, but is never part of ``live_in`` (it does not
  arrive from a predecessor).
* A phi source variable is live exactly on the edge from its bound
  predecessor to the phi block (not on the other incoming edges, and not as a
  normal boundary value).
* Parameters are live on entry to the entry block.

Fixpoint (backward may-analysis):

    after_phi(B) = reverse transfer through B's real instructions from live_out
    live_in(B)   = after_phi(B) - {phi destinations of B}   (+ params at entry)
    edge(B,S)    = live_in(S) | {phi sources bound to edge (B,S)}
    live_out(B)  = union over successors S of edge(B,S)
"""

from __future__ import annotations

from .model import Block, Liveness, Program


def _reverse_transfer(block: Block, out_set: frozenset[str]):
    """Walk real instructions backwards from ``out_set``.

    Returns ``(live_before_first, live_after_per_instr)`` where
    ``live_before_first`` is liveness *after* the (separate) phi definitions.
    Rule per instruction: before = (after - defs) | uses.
    """
    cur = set(out_set)
    afters: list[frozenset[str]] = [frozenset()] * len(block.instrs)
    for i in range(len(block.instrs) - 1, -1, -1):
        ins = block.instrs[i]
        afters[i] = frozenset(cur)
        cur.difference_update(ins.defs)
        cur.update(ins.uses)
    return frozenset(cur), afters


def compute_liveness(program: Program) -> Liveness:
    blocks = {b.id: b for b in program.blocks}
    order = list(blocks)  # program block order; result is order-independent

    phi_dsts: dict[str, frozenset[str]] = {}
    phi_srcs_on_edge: dict[tuple[str, str], set[str]] = {}
    for b in program.blocks:
        phi_dsts[b.id] = frozenset(p.dst for p in b.phis)
        for p in b.phis:
            for pred, src in p.srcs.items():
                phi_srcs_on_edge.setdefault((pred, b.id), set()).add(src)

    succ = {b.id: tuple(b.succs) for b in program.blocks}

    # live-in excluding phi destinations (used across edges)
    in_edge: dict[str, frozenset[str]] = {bid: frozenset() for bid in order}
    out_set: dict[str, frozenset[str]] = {bid: frozenset() for bid in order}
    # liveness right after the phi group (used for per-instruction points)
    after_phi: dict[str, frozenset[str]] = {bid: frozenset() for bid in order}

    changed = True
    while changed:
        changed = False
        for bid in reversed(order):
            b = blocks[bid]
            edge_union: set[str] = set()
            for s in succ[bid]:
                edge_union |= in_edge[s]
                edge_union |= phi_srcs_on_edge.get((bid, s), set())
            new_out = frozenset(edge_union)
            new_after_phi, _ = _reverse_transfer(b, new_out)
            # Parameters have no defining instruction: uses that reach the
            # entry boundary keep them live; a never-used parameter is dead
            # and correctly absent here.
            new_in = new_after_phi - phi_dsts[bid]
            if new_out != out_set[bid] or new_in != in_edge[bid]:
                out_set[bid] = new_out
                in_edge[bid] = new_in
                after_phi[bid] = new_after_phi
                changed = True

    # Precise per-instruction points at the fixpoint. ``before`` of the first
    # real instruction includes phi destinations (the phis have executed at
    # block entry); later ``before`` sets follow the plain gen/kill walk.
    points: dict[tuple[str, int], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for bid in order:
        b = blocks[bid]
        _, afters = _reverse_transfer(b, out_set[bid])
        cur = set(after_phi[bid])
        for i, ins in enumerate(b.instrs):
            before = frozenset(cur)
            after = afters[i]
            points[(bid, i)] = (tuple(sorted(before)), tuple(sorted(after)))
            cur = set(after)  # live after instr i == live before instr i+1

    edge_live: dict[tuple[str, str], tuple[str, ...]] = {}
    for bid in order:
        for s in succ[bid]:
            live = set(in_edge[s]) | phi_srcs_on_edge.get((bid, s), set())
            edge_live[(bid, s)] = tuple(sorted(live))

    live_in = {bid: tuple(sorted(in_edge[bid])) for bid in order}
    live_out = {bid: tuple(sorted(out_set[bid])) for bid in order}

    return Liveness(edge_live=edge_live, live_in=live_in, live_out=live_out, points=points)
