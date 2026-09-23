"""V2 lowering: turn a view coloring + spill plans into register-level CFG.

Output IR (:class:`AllocatedProgram2`) is phi-free and contains only::

    op     reads=(view,..) writes=(view,..) clobbers=(view,..) [may_throw]
    copy   dst_view <- src_view
    load   dst_view <- [slot_offset]
    store  [slot_offset] <- src_view

Phi semantics are materialised on the corresponding predecessor edge —
normal edges exactly like v1 (tail append or critical-edge split);
exceptional edges ALWAYS get a dedicated split block, because the
exceptional transition leaves the source block in the middle (at the
may_throw instruction) and must not share the normal-path tail code.

Parallel-copy groups are sequentialised with a bit-level "still needed"
relation (a write clobbers every read whose bit range overlaps — aliasing
views included).  Register cycles are broken by

1. a genuinely free temporary view: wide enough, not aliasing ANY value
   live on that edge and not aliasing any pending action — never a
   still-live alias location; or, when no such view exists,
2. the dedicated scratch stack slot.

Which mechanism fired is recorded per edge in
``AllocatedProgram2.edge_breaks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .allocator2 import EdgePlan2, Slot, VirtualProgram2
from .machine import Machine
from .model import Program


@dataclass(frozen=True)
class MI2:
    """One machine-level instruction (views are named)."""

    kind: str                 # "op" | "copy" | "load" | "store"
    dst: str = ""
    src: str = ""
    slot: int = -1
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    clobbers: tuple[str, ...] = ()
    uid: int = -1
    note: str = ""
    may_throw: bool = False

    def render(self) -> str:
        if self.kind == "op":
            rd = ",".join(self.reads)
            wr = ",".join(self.writes)
            cl = f" clobber[{','.join(self.clobbers)}]" if self.clobbers else ""
            th = " may_throw" if self.may_throw else ""
            tag = f" ; {self.note}" if self.note else ""
            return f"op{self.uid:<4d} {wr} = f({rd}){cl}{th}{tag}"
        if self.kind == "copy":
            return f"copy {self.dst} <- {self.src}"
        if self.kind == "load":
            return f"load {self.dst} <- [slot{self.slot}]"
        if self.kind == "store":
            return f"store [slot{self.slot}] <- {self.src}"
        raise AssertionError(self.kind)


@dataclass
class ABlock2:
    label: str
    instrs: list[MI2] = field(default_factory=list)
    succs: list[int] = field(default_factory=list)
    exc_succs: list[int] = field(default_factory=list)
    origin: int = -1
    split_edge: tuple[int, int] | None = None
    split_kind: str = ""      # "normal" | "exc"


@dataclass
class AllocatedProgram2:
    blocks: tuple[ABlock2, ...]
    machine: Machine
    slots: dict[int, Slot]
    scratch_slot: Slot
    assignment: dict[int, str]
    param_views: dict[int, str]
    edge_breaks: dict[tuple[int, int], tuple[str, ...]]
    frame_size: int

    def render(self) -> str:
        lines: list[str] = []
        for i, b in enumerate(self.blocks):
            tail = ""
            if b.succs:
                tail += " -> " + ",".join(self.blocks[s].label for s in b.succs)
            if b.exc_succs:
                tail += " ~~> " + ",".join(self.blocks[s].label for s in b.exc_succs)
            tag = ""
            if b.split_edge is not None:
                p, q = b.split_edge
                tag = f"  ; split of {'exceptional ' if b.split_kind == 'exc' else ''}edge {p}->{q}"
            lines.append(f"{b.label}:{tail}{tag}")
            for mi in b.instrs:
                lines.append(f"    {mi.render()}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# parallel-copy sequentialisation (bit-level)
# --------------------------------------------------------------------------


def _write_locs(machine: Machine, e: list):
    """Locations written by a pending action: ("r", group, lo, hi) or ("m", off)."""
    if e[0] == "copy":
        v = machine.view(e[1])
        return [("r", v.group, v.lo, v.hi)]
    if e[0] == "store":
        return [("m", e[3])]
    if e[0] == "load":
        v = machine.view(e[1])
        return [("r", v.group, v.lo, v.hi)]
    raise AssertionError(e[0])


def _read_locs(machine: Machine, e: list):
    if e[0] == "copy":
        v = machine.view(e[2])
        return [("r", v.group, v.lo, v.hi)]
    if e[0] == "store":
        v = machine.view(e[2])
        return [("r", v.group, v.lo, v.hi)]
    if e[0] == "load":
        return [("m", e[3])]
    raise AssertionError(e[0])


def _locs_overlap(a, b) -> bool:
    for x in a:
        for y in b:
            if x[0] != y[0]:
                continue
            if x[0] == "m":
                if x[1] == y[1]:
                    return True
            elif x[1] == y[1] and x[2] < y[3] and y[2] < x[3]:
                return True
    return False


def _find_free_temp(
    machine: Machine, width: int, blocked_ranges: list
) -> str | None:
    """First view (machine order) of width >= ``width`` not overlapping any
    blocked bit range.  ``blocked_ranges`` holds ("r", group, lo, hi)."""
    for v in machine.views:
        if v.width < width:
            continue
        rng = ("r", v.group, v.lo, v.hi)
        if any(
            b[1] == rng[1] and b[2] < rng[3] and rng[2] < b[3]
            for b in blocked_ranges
            if b[0] == "r"
        ):
            continue
        return v.name
    return None


def sequential_parallel2(
    actions: list[list],
    machine: Machine,
    occupied_views: set[str],
    scratch: Slot,
) -> tuple[list[MI2], tuple[str, ...]]:
    """Sequentialise a parallel assignment of views and memory slots.

    ``actions`` items are ``["copy", dst_view, src_view]`` or
    ``["store", slot_off, src_view]``.  ``occupied_views`` are views holding
    values live at this point (their bits must not be borrowed).  Returns
    the machine instructions plus, per broken register cycle, the mechanism
    used (``"temp"`` for a genuinely free temporary view, ``"slot"`` for
    the scratch stack slot).
    """
    pending: list[list] = []
    for a in actions:
        kind = a[0]
        if kind == "copy":
            _, dst, src = a
            if dst == src:
                continue  # coalesced away
            pending.append(["copy", dst, src, -1])
        elif kind == "store":
            _, slot, src = a
            pending.append(["store", -1, src, slot])
        else:  # pragma: no cover
            raise AssertionError(kind)

    out: list[MI2] = []
    breaks: list[str] = []

    def emit(e: list) -> None:
        if e[0] == "copy":
            out.append(MI2(kind="copy", dst=e[1], src=e[2]))
        elif e[0] == "store":
            out.append(MI2(kind="store", src=e[2], slot=e[3]))
        else:
            out.append(MI2(kind="load", dst=e[1], slot=e[3]))

    while pending:
        move = None
        for e in pending:
            writes = _write_locs(machine, e)
            if not any(
                _locs_overlap(writes, _read_locs(machine, other))
                for other in pending
                if other is not e
            ):
                move = e
                break
        if move is not None:
            pending.remove(move)
            emit(move)
            continue

        # only register permutation cycles remain: break one
        cyc = next(e for e in pending if e[0] == "copy")
        saved = cyc[1]  # view whose old value is still needed
        saved_view = machine.view(saved)
        tail = next(
            e
            for e in pending
            if e is not cyc
            and _locs_overlap(_read_locs(machine, e), _write_locs(machine, cyc))
        )
        blocked = [("r", saved_view.group, saved_view.lo, saved_view.hi)]
        for vw in occupied_views:
            v = machine.view(vw)
            blocked.append(("r", v.group, v.lo, v.hi))
        for e in pending:
            blocked.extend(_write_locs(machine, e))
            blocked.extend(_read_locs(machine, e))
        temp = _find_free_temp(machine, saved_view.width, blocked)
        if temp is not None:
            out.append(MI2(kind="copy", dst=temp, src=saved))
            tail[2] = temp
            breaks.append("temp")
        else:
            out.append(MI2(kind="store", src=saved, slot=scratch.offset))
            tail[0] = "load"
            tail[2] = -1
            tail[3] = scratch.offset
            breaks.append("slot")
    return out, tuple(breaks)


# --------------------------------------------------------------------------
# lowering
# --------------------------------------------------------------------------


def lower_program2(
    program: Program,
    machine: Machine,
    vp: VirtualProgram2,
    color: dict[int, str],
    report,
    slots: dict[int, Slot],
    scratch: Slot,
) -> AllocatedProgram2:
    n = len(program.blocks)
    blocks = [
        ABlock2(
            label=b.label,
            succs=list(b.succs),
            exc_succs=list(b.exc_succs),
            origin=i,
        )
        for i, b in enumerate(program.blocks)
    ]

    # ---- instruction bodies ----------------------------------------------
    for i, b in enumerate(program.blocks):
        for ins in b.instrs:
            plan = vp.instr_plans[ins.uid]
            for kind, a, v in plan.pre:
                if kind == "load":
                    blocks[i].instrs.append(
                        MI2(kind="load", dst=color[a], slot=slots[v].offset)
                    )
                else:
                    if color[a] != color[v]:
                        blocks[i].instrs.append(
                            MI2(kind="copy", dst=color[a], src=color[v])
                        )
            if ins.is_call:
                clobbers = machine.clobbered_by_call(ins.clobber_views)
            else:
                clobbers = tuple(dict.fromkeys(ins.clobber_views))
            blocks[i].instrs.append(
                MI2(
                    kind="op",
                    reads=tuple(color[r] for r in plan.reads),
                    writes=tuple(color[w] for w in plan.writes),
                    clobbers=clobbers,
                    uid=ins.uid,
                    note=ins.note,
                    may_throw=ins.may_throw,
                )
            )
            for kind, a, v in plan.post:
                if kind == "store":
                    blocks[i].instrs.append(
                        MI2(
                            kind="store",
                            src=color[a],
                            slot=slots[v].offset,
                        )
                    )
                else:
                    if color[a] != color[v]:
                        blocks[i].instrs.append(
                            MI2(
                                kind="copy",
                                dst=color[a],
                                src=color[v],
                            )
                        )

    # ---- edge actions ------------------------------------------------------
    edge_breaks: dict[tuple[int, int], tuple[str, ...]] = {}

    def emit_edge_actions(mi_list: list[MI2], ep: EdgePlan2) -> None:
        for kind, t, svar in ep.loads:
            mi_list.append(MI2(kind="load", dst=color[t], slot=slots[svar].offset))
        actions: list[list] = []
        for kind, d, s in ep.copies:
            actions.append(["copy", color[d], color[s]])
        for kind, s, dvar in ep.stores:
            actions.append(["store", slots[dvar].offset, color[s]])
        occupied = {
            color[v]
            for v in report.edge_live[(ep.pred, ep.succ)]
            if v not in vp.spilled and v in color
        }
        mis, breaks = sequential_parallel2(actions, machine, occupied, scratch)
        mi_list.extend(mis)
        if breaks:
            edge_breaks[(ep.pred, ep.succ)] = breaks

    extra: list[ABlock2] = []
    for (p, q), ep in vp.edge_plans.items():
        empty = not ep.loads and not ep.copies and not ep.stores
        if ep.kind == "normal":
            if empty:
                continue
            if len(program.blocks[p].succs) == 1:
                emit_edge_actions(blocks[p].instrs, ep)
            else:
                split = ABlock2(
                    label=f"{blocks[p].label}__split_{blocks[q].label}",
                    succs=[q],
                    origin=-1,
                    split_edge=(p, q),
                    split_kind="normal",
                )
                sid = n + len(extra)
                extra.append(split)
                blocks[p].succs[blocks[p].succs.index(q)] = sid
                emit_edge_actions(split.instrs, ep)
        else:
            # exceptional edges always get a dedicated split block: the
            # exceptional transition leaves the source block mid-body
            idx = program.blocks[p].exc_succs.index(q)
            if empty:
                blocks[p].exc_succs[idx] = q
                continue
            split = ABlock2(
                label=f"{blocks[p].label}__xsplit_{blocks[q].label}",
                succs=[q],
                origin=-1,
                split_edge=(p, q),
                split_kind="exc",
            )
            sid = n + len(extra)
            extra.append(split)
            blocks[p].exc_succs[idx] = sid
            emit_edge_actions(split.instrs, ep)

    all_blocks = tuple(blocks + extra)
    assignment = {v: color[v] for v in sorted(vp.real_nodes) if v in color}
    return AllocatedProgram2(
        blocks=all_blocks,
        machine=machine,
        slots=slots,
        scratch_slot=scratch,
        assignment=assignment,
        param_views={v: assignment[v] for v in program.params if v in assignment},
        edge_breaks=edge_breaks,
        frame_size=scratch.end,
    )
