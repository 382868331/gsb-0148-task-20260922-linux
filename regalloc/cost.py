"""Weighted spill-memory cost: heuristic estimates and *actual* accounting.

Two strictly separated quantities live here:

1. :func:`spill_cost_table` — a **heuristic estimate**, computed directly from
   the source program's real use/def points and its edge-bound phi uses.  For
   every candidate variable ``v`` it predicts how many reload/store actions
   spilling *that variable* would insert, and weights each action by the
   execution frequency of the block (or predecessor edge) it sits on:

   * a use of ``v`` in a body instruction  -> one reload,  weight ``freq[block]``
     (this also covers fixed-register uses of a spilled value, which get a
     reload plus a register copy — the copy is not memory traffic);
   * a def of ``v`` in a body instruction  -> one store,   weight ``freq[block]``
     (fixed-register defs of spilled values are stored straight from the
     fixed pseudo register — still exactly one store);
   * on every incoming edge of a block whose phi **defines** ``v`` -> one
     store on that edge, weighted by the *predecessor* block frequency;
   * on every edge on which ``v`` is a phi **source** -> one reload on that
     edge, weighted by the *predecessor* frequency;
   * a spilled parameter is born in its slot at entry: no entry traffic.

   The estimate is independent of which other variables are spilled and of
   the eventual coloring (coalescing only removes register *copies*; it can
   never remove a load/store).  The one memory effect it deliberately does
   not predict is the scratch-slot store/load pair used to break a register
   permutation cycle during phi lowering, because that depends on the final
   colors.  Those pairs are rare and are counted in the *actual* stats.

2. :func:`actual_access_stats` — the ground truth: it walks the final
   lowered machine code and counts the load/store instructions that really
   were emitted (scratch-slot cycle breakers included), weighted by the
   frequency of the lowered block they execute in (split edge blocks inherit
   their predecessor edge's frequency).  Demonstrated savings are always
   reported from THIS function, never from the heuristic.

Frequencies are non-negative integers supplied by the caller.  They influence
spill selection only; semantics never depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VarSpillCost:
    """Heuristic weighted reload/store events predicted for spilling one var.

    ``loads``/``stores`` are unweighted event counts; the weighted fields
    multiply each event by its block/predecessor frequency.
    """

    loads: int = 0
    stores: int = 0
    weighted_loads: int = 0
    weighted_stores: int = 0

    @property
    def events(self) -> int:
        return self.loads + self.stores

    @property
    def weighted_total(self) -> int:
        return self.weighted_loads + self.weighted_stores


@dataclass(frozen=True)
class AccessStats:
    """Actually emitted memory traffic in the lowered program.

    All fields describe real ``load``/``store`` machine instructions, counted
    from the final allocation — never heuristic estimates.

    * ``loads`` / ``stores``: raw instruction counts;
    * ``weighted_loads`` / ``weighted_stores``: same instructions weighted by
      the execution frequency of the lowered block they appear in (equal to
      the raw counts when no frequencies were supplied, i.e. every block has
      implicit weight 1);
    * ``by_slot``: per-spill-slot breakdown
      ``(slot, loads, stores, weighted_loads, weighted_stores)`` sorted by
      slot; the phi-lowering scratch slot appears under its own (larger) id
      when a register cycle had to be broken through memory.
    """

    loads: int
    stores: int
    weighted_loads: int
    weighted_stores: int
    by_slot: tuple[tuple[int, int, int, int, int], ...] = ()

    @property
    def total(self) -> int:
        return self.loads + self.stores

    @property
    def weighted_total(self) -> int:
        return self.weighted_loads + self.weighted_stores

    def row(self) -> str:
        return (
            f"loads={self.loads} stores={self.stores} "
            f"weighted(loads={self.weighted_loads}, stores={self.weighted_stores}, "
            f"total={self.weighted_total})"
        )


def spill_cost_table(program, frequencies: tuple[int, ...]) -> dict[int, VarSpillCost]:
    """Predict per-variable spill traffic from real use/def and edge phi uses.

    ``frequencies`` must be one non-negative int per block.  The scan mirrors
    the spill-form construction in :mod:`regalloc.allocator` exactly (see
    module docstring for the event rules), but is independent of its
    interference graph and coloring.
    """
    blocks = program.blocks
    freq = frequencies
    loads: dict[int, int] = {}
    stores: dict[int, int] = {}
    wloads: dict[int, int] = {}
    wstores: dict[int, int] = {}

    def bump(table: dict[int, int], v: int, amount: int) -> None:
        table[v] = table.get(v, 0) + amount

    # ---- body use/def points ---------------------------------------------
    for bi, b in enumerate(blocks):
        f = freq[bi]
        for ins in b.instrs:
            # each distinct use of a spilled value gets exactly one reload;
            # operand tuples are unique-checked at construction time
            for u in ins.uses:
                bump(loads, u, 1)
                bump(wloads, u, f)
            # each def gets exactly one store after the instruction
            for d in ins.defs:
                bump(stores, d, 1)
                bump(wstores, d, f)

    # ---- edge-bound phi uses ---------------------------------------------
    # A phi def d of block b is materialised once per incoming edge (p -> b):
    # the incoming value lands in d's slot, i.e. one store attributed to d;
    # when the edge source s is itself spilled it is reloaded first.  Edge
    # actions execute on the predecessor's frequency.
    for b in blocks:
        for phi in b.phis:
            d = phi.def_var
            for p, s in phi.sources:
                f = freq[p]
                # d spilled  -> its slot is written on this edge (one store)
                bump(stores, d, 1)
                bump(wstores, d, f)
                # s spilled  -> it is read out of its slot on this edge
                # (one load; even when s == d the spill form reloads the
                # slot into a temp and writes it straight back)
                bump(loads, s, 1)
                bump(wloads, s, f)

    table: dict[int, VarSpillCost] = {}
    for v in set(loads) | set(stores):
        table[v] = VarSpillCost(
            loads=loads.get(v, 0),
            stores=stores.get(v, 0),
            weighted_loads=wloads.get(v, 0),
            weighted_stores=wstores.get(v, 0),
        )
    return table


def lowered_block_frequencies(program, allocated_program) -> tuple[int, ...] | None:
    """Frequency of each lowered block.

    Origin blocks keep their source frequency; a block inserted to split a
    critical edge executes exactly when its edge does, so it inherits the
    *predecessor* block's frequency.  Returns ``None`` when the program
    carries no frequency hints.
    """
    if program.frequencies is None:
        return None
    freq = program.frequencies
    out: list[int] = []
    for b in allocated_program.blocks:
        if b.origin >= 0:
            out.append(freq[b.origin])
        else:
            p, _q = b.split_edge
            out.append(freq[p])
    return tuple(out)


def actual_access_stats(program, allocation) -> AccessStats:
    """Count the load/store instructions actually present in the final code.

    This is the ground-truth accounting used for all reported savings; it
    never consults the spill heuristic.
    """
    ap = allocation.allocated_program
    block_freq = lowered_block_frequencies(program, ap)

    loads = 0
    stores = 0
    wloads = 0
    wstores = 0
    per_slot: dict[int, list[int]] = {}

    def slot_row(slot: int) -> list[int]:
        return per_slot.setdefault(slot, [0, 0, 0, 0])  # l, s, wl, ws

    for bi, b in enumerate(ap.blocks):
        f = 1 if block_freq is None else block_freq[bi]
        for mi in b.instrs:
            if mi.kind == "load":
                loads += 1
                wloads += f
                row = slot_row(mi.slot)
                row[0] += 1
                row[2] += f
            elif mi.kind == "store":
                stores += 1
                wstores += f
                row = slot_row(mi.slot)
                row[1] += 1
                row[3] += f

    by_slot = tuple(
        (slot, l, s, wl, ws)
        for slot, (l, s, wl, ws) in sorted(per_slot.items())
    )
    return AccessStats(
        loads=loads,
        stores=stores,
        weighted_loads=wloads,
        weighted_stores=wstores,
        by_slot=by_slot,
    )
