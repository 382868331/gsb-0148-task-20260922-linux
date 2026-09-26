"""Stack-slot layout after register allocation and spill lowering.

The register allocator first gives every spilled value a distinct logical
slot.  This module rebuilds liveness from the actual inserted memory events.
Loads and stores around ordinary instructions remain in their block, while
phi spill loads/stores are placed on their corresponding CFG edges.  Values
whose memory lifetimes do not overlap may share a byte interval.
"""

from __future__ import annotations

from dataclasses import dataclass

from .allocator import build_virtual_program
from .model import ProgramFormatError, check_int


@dataclass(frozen=True)
class StackAccess:
    block: int
    index: int
    kind: str
    variable: int
    register: int
    offset: int
    edge: tuple[int, int] | None = None


@dataclass(frozen=True)
class StackLayout:
    frame_size: int
    frame_alignment: int
    offsets: dict[int, int]
    sizes: dict[int, int]
    alignments: dict[int, int]
    accesses: tuple[StackAccess, ...]
    interferences: frozenset[tuple[int, int]]
    optimal: bool
    budget_exhausted: bool
    placements_checked: int
    live_in: tuple[frozenset[int], ...]
    live_out: tuple[frozenset[int], ...]
    synthetic_blocks: tuple[tuple[str, tuple[int, int]], ...]


def _power_of_two(value: object, path: str) -> int:
    value = check_int(value, path, minimum=1)
    if value & (value - 1):
        raise ProgramFormatError(f"{path}: alignment must be a power of two, got {value}")
    return value


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _bytes_overlap(a: int, a_size: int, b: int, b_size: int) -> bool:
    return a < b + b_size and b < a + a_size


def _stack_liveness(program, allocation):
    spilled = set(allocation.slots)
    vp = build_virtual_program(program, spilled)
    ap = allocation.allocated_program

    events: list[list[tuple[tuple[int, ...], tuple[int, ...]]]] = []
    meta: list[tuple[str, tuple[int, int]]] = []
    succs: list[list[int]] = []

    for bi, block in enumerate(program.blocks):
        meta.append((f"body:{bi}", (bi, bi)))
        succs.append(list(block.succs))
        block_events: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for ins in block.instrs:
            plan = vp.instr_plans[ins.uid]
            uses = tuple(v for kind, _tmp, v in plan.pre if kind == "load")
            defs = tuple(v for kind, _tmp, v in plan.post if kind == "store")
            if uses or defs:
                block_events.append((uses, defs))
        events.append(block_events)

    origin_to_allocated = {block.origin: i for i, block in enumerate(ap.blocks) if block.origin >= 0}
    split_by_edge = {block.split_edge: i for i, block in enumerate(ap.blocks) if block.split_edge is not None}

    for bi, block in enumerate(program.blocks):
        for succ in sorted(block.succs):
            node = len(events)
            succs[bi] = [node if x == succ else x for x in succs[bi]]
            meta.append((f"edge:{bi}->{succ}", (bi, succ)))
            succs.append([succ])
            edge_events: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
            allocated_index = split_by_edge.get((bi, succ), origin_to_allocated[bi])
            edge_block = ap.blocks[allocated_index]
            edge_mis = list(edge_block.instrs)
            if allocated_index == origin_to_allocated[bi]:
                last_op = max((i for i, mi in enumerate(edge_mis) if mi.kind == "op"), default=-1)
                edge_mis = edge_mis[last_op + 1 :]
            for mi in edge_mis:
                if mi.kind == "load":
                    edge_events.append(((mi.slot,), ()))
                elif mi.kind == "store":
                    edge_events.append(((), (mi.slot,)))
            events.append(edge_events)

    preds: list[list[int]] = [[] for _ in events]
    for u, outs in enumerate(succs):
        for v in outs:
            preds[v].append(u)

    upward: list[set[int]] = []
    killed: list[set[int]] = []
    for node_events in events:
        node_kill: set[int] = set()
        node_up: set[int] = set()
        for uses, defs in node_events:
            node_up.update(v for v in uses if v not in node_kill)
            node_kill.update(defs)
        upward.append(node_up)
        killed.append(node_kill)

    live_in = [set() for _ in events]
    live_out = [set() for _ in events]
    changed = True
    while changed:
        changed = False
        for i in range(len(events) - 1, -1, -1):
            out: set[int] = set()
            for succ in succs[i]:
                out |= live_in[succ]
            new_in = upward[i] | (out - killed[i])
            if out != live_out[i] or new_in != live_in[i]:
                live_out[i] = out
                live_in[i] = new_in
                changed = True

    live_in[0].update(v for v in program.params if v in spilled)

    interferences: set[tuple[int, int]] = set()

    def clique(values) -> None:
        ordered = sorted(set(values))
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                interferences.add((a, b))

    for i, node_events in enumerate(events):
        clique(live_in[i])
        clique(live_out[i])
        live = set(live_out[i])
        for uses, defs in reversed(node_events):
            clique(live | set(uses))
            live = (live - set(defs)) | set(uses)
            clique(live)

    scratch_used = any(
        mi.slot == ap.scratch_slot
        for block in ap.blocks
        for mi in block.instrs
        if mi.kind in ("load", "store")
    )
    return spilled, frozenset(interferences), live_in, live_out, meta, scratch_used, vp


def _normalise_dimensions(values, sizes, alignments):
    if isinstance(sizes, bool) or not isinstance(sizes, (int, dict)):
        raise ProgramFormatError("sizes: expected positive int or {variable: int}")
    if isinstance(alignments, bool) or not isinstance(alignments, (int, dict)):
        raise ProgramFormatError("alignments: expected positive power-of-two int or mapping")
    size_map = {v: sizes for v in values} if isinstance(sizes, int) else dict(sizes)
    align_map = (
        {v: alignments for v in values}
        if isinstance(alignments, int)
        else dict(alignments)
    )
    out_sizes = {v: check_int(size_map.get(v), f"sizes[{v}]", minimum=1) for v in sorted(values)}
    out_aligns = {
        v: _power_of_two(align_map.get(v), f"alignments[{v}]") for v in sorted(values)
    }
    return out_sizes, out_aligns


def _greedy(order, sizes, aligns, adj, frame_alignment):
    offsets: dict[int, int] = {}
    for v in order:
        candidates = [0]
        for u, base in offsets.items():
            candidates.append(_align_up(base + sizes[u], aligns[v]))
        candidates.sort()
        for offset in candidates:
            if all(
                u not in offsets or not _bytes_overlap(offset, sizes[v], offsets[u], sizes[u])
                for u in adj[v]
            ):
                offsets[v] = offset
                break
        else:  # pragma: no cover
            raise AssertionError("internal: greedy stack placement failed")
    end = max(_align_up(offsets[v] + sizes[v], frame_alignment) for v in offsets)
    return offsets, end


def layout_stack_slots(
    allocation,
    *,
    sizes,
    alignments,
    frame_alignment: int | None = None,
    budget: int = 10_000,
) -> StackLayout:
    """Assign final frame offsets to logical spill slots, in place."""
    program = allocation.program
    ap = allocation.allocated_program
    budget = check_int(budget, "budget", minimum=0)

    spilled, interferences, live_in, live_out, meta, scratch_used, vp = _stack_liveness(program, allocation)
    objects = set(spilled)
    if scratch_used:
        objects.add(ap.scratch_slot)

    empty = StackLayout(
        0, 1, {}, {}, {}, (), frozenset(), True, False, 0,
        tuple(frozenset(x) for x in live_in),
        tuple(frozenset(x) for x in live_out),
        tuple(meta),
    )
    allocation.stack_layout = empty
    if not objects:
        return empty

    size_map, align_map = _normalise_dimensions(spilled, sizes, alignments)
    if scratch_used:
        size_map[ap.scratch_slot] = max(8, *size_map.values())
        align_map[ap.scratch_slot] = max(8, *align_map.values())
    if frame_alignment is None:
        frame_alignment = max(align_map.values())
    frame_alignment = _power_of_two(frame_alignment, "frame_alignment")

    adj = {v: set() for v in objects}
    for a, b in interferences:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    order = sorted(objects, key=lambda v: (-(len(adj[v]) * size_map[v]), -size_map[v], v))
    best_offsets, best_size = _greedy(order, size_map, align_map, adj, frame_alignment)

    subset_sums = {0}
    if len(objects) <= 20:
        for size in size_map.values():
            subset_sums.update({x + size for x in subset_sums if x + size < best_size})

    current: dict[int, int] = {}
    checked = 0
    exhausted = False

    def current_end() -> int:
        return _align_up(max(current[v] + size_map[v] for v in current), frame_alignment)

    def search(pos: int) -> bool:
        nonlocal checked, exhausted, best_offsets, best_size
        if pos == len(order):
            end = current_end()
            if end < best_size:
                best_size = end
                best_offsets = dict(current)
            return True
        checked += 1
        if checked > budget:
            exhausted = True
            return False

        v = order[pos]
        if len(objects) <= 20:
            raw = subset_sums
        else:
            raw = {0}
            for u, base in current.items():
                raw.add(base + size_map[u])
        candidates = sorted({_align_up(x, align_map[v]) for x in raw if x < best_size})
        found = False
        for offset in candidates:
            if _align_up(offset + size_map[v], frame_alignment) >= best_size:
                continue
            if any(
                u in current and _bytes_overlap(offset, size_map[v], current[u], size_map[u])
                for u in adj[v]
            ):
                continue
            current[v] = offset
            if current_end() < best_size:
                found = search(pos + 1) or found
            del current[v]
            if exhausted:
                return found
        return found

    search(0)

    edge_slots: dict[tuple[int, int], set[int]] = {}
    for (pred, succ), plan in vp.edge_plans.items():
        slots = {var for _kind, _tmp, var in plan.loads}
        slots.update(var for _kind, _tmp, var in plan.stores)
        edge_slots[(pred, succ)] = slots

    accesses: list[StackAccess] = []
    for bi, block in enumerate(ap.blocks):
        edge = block.split_edge
        last_op = max((i for i, mi in enumerate(block.instrs) if mi.kind == "op"), default=-1)
        if edge is None and len(block.succs) == 1:
            target = block.succs[0]
            target_origin = ap.blocks[target].origin
            if target_origin >= 0:
                origin = block.origin if block.origin >= 0 else None
                if origin is not None:
                    unsplit_edge = (origin, target_origin)
                else:
                    unsplit_edge = None
            else:
                unsplit_edge = None
        else:
            unsplit_edge = None
        for ii, mi in enumerate(block.instrs):
            if mi.kind not in ("load", "store"):
                continue
            offset = best_offsets[mi.slot]
            object.__setattr__(mi, "offset", offset)
            register = mi.dst if mi.kind == "load" else mi.src
            source_edge = unsplit_edge
            if edge is not None:
                access_edge = edge
            elif ii > last_op and source_edge in edge_slots and mi.slot in edge_slots[source_edge]:
                access_edge = source_edge
            else:
                access_edge = None
            accesses.append(StackAccess(bi, ii, mi.kind, mi.slot, register, offset, access_edge))

    layout = StackLayout(
        frame_size=best_size,
        frame_alignment=frame_alignment,
        offsets=dict(sorted(best_offsets.items())),
        sizes=dict(sorted(size_map.items())),
        alignments=dict(sorted(align_map.items())),
        accesses=tuple(accesses),
        interferences=frozenset(interferences),
        optimal=not exhausted,
        budget_exhausted=exhausted,
        placements_checked=checked,
        live_in=tuple(frozenset(x) for x in live_in),
        live_out=tuple(frozenset(x) for x in live_out),
        synthetic_blocks=tuple(meta),
    )
    allocation.stack_layout = layout
    return layout
