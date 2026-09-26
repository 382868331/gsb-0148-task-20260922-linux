"""Stack-slot layout: pack non-overlapping spill lifetimes into one frame.

The register allocator gives every spilled value its own *logical* slot id
(the variable id).  This stage runs **after** spill code has been inserted
and phi copies have been materialised on their predecessor edges, and:

1. recomputes liveness on the lowered, phi-free machine CFG.  Phi reads and
   writes have already become loads/stores placed on the correct edge (in a
   predecessor tail or a critical-edge split block), so a plain backward
   liveness analysis over memory slots is sufficient and cannot mistake an
   edge action for an ordinary in-block instruction;
2. builds an interference graph over logical slots: two slots interfere if
   both hold a live value at any program point (their occupied byte ranges
   must then be disjoint);
3. lays the slots out in the caller-given sizes/alignments, minimising the
   final frame size by exhaustive, budgeted enumeration.  If the budget runs
   out before optimality is proven, a legal greedy layout is returned with
   proven_optimal set to False.

A value spilled to memory is not destroyed by a call (only registers are
clobbered), so a slot may be reused across a call whenever ordinary liveness
says the old value is dead; liveness alone keeps still-needed values safe.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LAYOUT_BUDGET = 200_000


@dataclass(frozen=True)
class SlotLiveness:
    '''Slot liveness on the lowered CFG (indexed by allocated block).'''

    live_in: tuple[frozenset[int], ...]
    live_out: tuple[frozenset[int], ...]
    instr_before: tuple[tuple[frozenset[int], ...], ...]
    instr_after: tuple[tuple[frozenset[int], ...], ...]


def slot_liveness(allocated_program, spilled_params) -> SlotLiveness:
    '''Backward liveness of memory slots over the lowered MI CFG.

    A store defines (kills) the slot; a load uses it.  Spilled (stack-passed)
    parameters live in their slot from function entry, so their slot is
    live-in to the entry block.
    '''
    blocks = allocated_program.blocks
    n = len(blocks)
    seed = frozenset(spilled_params)
    live_in: list[set[int]] = [set(seed) if i == 0 else set() for i in range(n)]
    live_out: list[set[int]] = [set() for _ in range(n)]
    befores: list[list[frozenset[int]]] = [
        [frozenset() for _ in b.instrs] for b in blocks
    ]
    afters: list[list[frozenset[int]]] = [
        [frozenset() for _ in b.instrs] for b in blocks
    ]

    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            out: set[int] = set()
            for s in blocks[i].succs:
                out |= live_in[s]
            live = set(out)
            for j in range(len(blocks[i].instrs) - 1, -1, -1):
                mi = blocks[i].instrs[j]
                afters[i][j] = frozenset(live)
                if mi.kind == 'store':
                    live.discard(mi.slot)
                elif mi.kind == 'load':
                    live.add(mi.slot)
                befores[i][j] = frozenset(live)
            if i == 0:
                live |= seed
            if live != live_in[i] or out != live_out[i]:
                live_in[i] = live
                live_out[i] = out
                changed = True

    return SlotLiveness(
        live_in=tuple(frozenset(s) for s in live_in),
        live_out=tuple(frozenset(s) for s in live_out),
        instr_before=tuple(tuple(row) for row in befores),
        instr_after=tuple(tuple(row) for row in afters),
    )


def build_slot_interference(liveness: SlotLiveness) -> dict[int, set[int]]:
    '''Two slots interfere when simultaneously live at some instruction.

    Every per-instruction live-before set is made a clique: all values a
    machine instruction can observe coexist there.  A store target carries
    its new value next to every surviving value, so the conservative
    before-set clique also covers write-point coexistence.
    '''
    adj: dict[int, set[int]] = {}

    def add(v: int) -> None:
        adj.setdefault(v, set())

    for row in liveness.instr_before:
        for live in row:
            members = sorted(live)
            for v in members:
                add(v)
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    x, y = members[a], members[b]
                    adj[x].add(y)
                    adj[y].add(x)
    for s in liveness.live_in:
        for v in s:
            add(v)
    return adj


@dataclass(frozen=True)
class StackLayout:
    '''Final frame packing result.'''

    slot_offsets: dict[int, int]
    slot_sizes: dict[int, int]
    slot_alignments: dict[int, int]
    frame_size: int
    frame_alignment: int
    scratch_slot: int
    scratch_offset: int
    proven_optimal: bool
    budget: int
    budget_spent: int
    note: str

    def as_jsonable(self) -> dict:
        return {
            'slot_offsets': dict(sorted(self.slot_offsets.items())),
            'slot_sizes': dict(sorted(self.slot_sizes.items())),
            'frame_size': self.frame_size,
            'frame_alignment': self.frame_alignment,
            'proven_optimal': self.proven_optimal,
            'budget_spent': self.budget_spent,
            'note': self.note,
        }


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _fits(offset: int, size: int, placed: dict[int, int], sizes: dict[int, int],
          conflicts: dict[int, set[int]], slot: int) -> bool:
    for other in conflicts.get(slot, ()):
        if other not in placed:
            continue
        other_off = placed[other]
        if offset < other_off + sizes[other] and other_off < offset + size:
            return False
    return True


def _greedy(order: list[int], sizes: dict[int, int], aligns: dict[int, int],
            conflicts: dict[int, set[int]]) -> dict[int, int]:
    '''Deterministic legal packing: lowest aligned fitting offset.'''
    placed: dict[int, int] = {}
    for v in order:
        size, align = sizes[v], aligns[v]
        candidate = 0
        while True:
            if candidate % align == 0 and _fits(
                candidate, size, placed, sizes, conflicts, v
            ):
                placed[v] = candidate
                break
            candidate += 1
    return placed


def pack_slots(*, logical_slots, scratch_slot: int, conflicts: dict[int, set[int]],
               sizes: dict[int, int], aligns: dict[int, int], frame_alignment: int,
               budget: int = DEFAULT_LAYOUT_BUDGET) -> StackLayout:
    '''Pack logical slots into a frame of minimum (budgeted) size.

    logical_slots contains every frame slot, including the scratch slot used
    to break register-permutation cycles.  The returned layout is always
    legal; proven_optimal is False only when the enumeration budget ran out.
    '''
    slots = sorted(logical_slots)
    for v in slots:
        conflicts.setdefault(v, set())
    # Large, strongly constrained slots first: prunes early while remaining
    # complete (every legal offset combination is still enumerated).
    order = sorted(slots, key=lambda v: (-sizes[v], -len(conflicts[v]), v))

    greedy = _greedy(order, sizes, aligns, conflicts)
    greedy_end = max((off + sizes[v] for v, off in greedy.items()), default=0)
    best = [_round_up(greedy_end, frame_alignment)]
    best_place = [dict(greedy)]
    spent = [0]
    timed_out = [False]
    placed: dict[int, int] = {}

    def frame_extent() -> int:
        end = max((off + sizes[v] for v, off in placed.items()), default=0)
        return _round_up(end, frame_alignment)

    def rec(depth: int) -> None:
        if timed_out[0]:
            return
        if depth == len(order):
            extent = frame_extent()
            if extent < best[0]:
                best[0] = extent
                best_place[0] = dict(placed)
            return
        v = order[depth]
        size, align = sizes[v], aligns[v]
        candidate = 0
        while candidate + size <= best[0]:
            if candidate % align == 0 and _fits(
                candidate, size, placed, sizes, conflicts, v
            ):
                spent[0] += 1
                if spent[0] > budget:
                    timed_out[0] = True
                    return
                placed[v] = candidate
                if frame_extent() <= best[0]:
                    rec(depth + 1)
                del placed[v]
                if timed_out[0]:
                    return
            candidate += 1

    rec(0)
    place = best_place[0]
    proven = not timed_out[0]
    if proven:
        note = 'exact minimum frame size proven by exhaustive offset enumeration'
    else:
        note = (
            f'layout budget ({budget}) exhausted before optimality was proven; '
            'a legal greedy layout is returned, not proven optimal'
        )
    return StackLayout(
        slot_offsets={v: place[v] for v in sorted(place)},
        slot_sizes={v: sizes[v] for v in sorted(place)},
        slot_alignments={v: aligns[v] for v in sorted(place)},
        frame_size=best[0],
        frame_alignment=frame_alignment,
        scratch_slot=scratch_slot,
        scratch_offset=place.get(scratch_slot, -1),
        proven_optimal=proven,
        budget=budget,
        budget_spent=min(spent[0], budget),
        note=note,
    )

