"""Lowering: turn a coloring + spill plans into register-level CFG code.

Output IR (:class:`AllocatedProgram`) is phi-free and contains only::

    op     reads=(r,..) writes=(r,..) clobbers=(r,..)
    copy   r_dst <- r_src
    load   r_dst <- slot
    store  slot <- r_src
    remat  r_dst = f(r,..)        # pure op replayed at the use point (uid tagged)

Phi semantics are materialised on the corresponding predecessor *edge*:

* if the predecessor has several successors (critical-edge case), a fresh
  split block is inserted and the edge actions live there;
* otherwise the actions are appended to the (unique-successor) predecessor.

Each edge action group is a parallel assignment of registers and slots; it is
sequentialised deterministically, and register-only cycles are broken with
one distinguished memory scratch slot (never a register, so the result is
correct for any K, including K=2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .allocator import EdgePlan, InstrPlan, VirtualProgram
from .model import Program


@dataclass(frozen=True)
class MI:
    """One machine-level instruction."""

    kind: str                 # "op" | "copy" | "load" | "store" | "remat"
    dst: int = -1
    src: int = -1
    slot: int = -1
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()
    clobbers: tuple[int, ...] = ()
    uid: int = -1
    note: str = ""

    def render(self) -> str:
        if self.kind == "op":
            rd = ",".join(f"r{x}" for x in self.reads)
            wr = ",".join(f"r{x}" for x in self.writes)
            cl = f" clobber[{','.join('r' + str(x) for x in self.clobbers)}]" if self.clobbers else ""
            tag = f" ; {self.note}" if self.note else ""
            return f"op{self.uid:<3d} {wr} = f({rd}){cl}{tag}"
        if self.kind == "copy":
            return f"copy r{self.dst} <- r{self.src}"
        if self.kind == "load":
            return f"load r{self.dst} <- slot{self.slot}"
        if self.kind == "store":
            return f"store slot{self.slot} <- r{self.src}"
        if self.kind == "remat":
            rd = ",".join(f"r{x}" for x in self.reads)
            return f"remat r{self.dst} = f({rd})  ; replay uid{self.uid}"
        raise AssertionError(self.kind)


@dataclass
class ABlock:
    label: str
    instrs: list[MI] = field(default_factory=list)
    succs: list[int] = field(default_factory=list)
    origin: int = -1            # original block index, -1 for split blocks
    split_edge: tuple[int, int] | None = None


@dataclass
class AllocatedProgram:
    blocks: tuple[ABlock, ...]
    k: int
    slots: dict[int, int]
    assignment: dict[int, int]
    rematerialized: dict[int, float]
    scratch_slot: int
    param_regs: dict[int, int]

    def render(self) -> str:
        lines: list[str] = []
        for i, b in enumerate(self.blocks):
            tail = " -> " + ",".join(self.blocks[s].label for s in b.succs)
            tag = ""
            if b.split_edge is not None:
                p, q = b.split_edge
                tag = f"  ; split of edge {p}->{q}"
            lines.append(f"{b.label}:{tail}{tag}")
            for mi in b.instrs:
                lines.append(f"    {mi.render()}")
        return "\n".join(lines)


def _sequential_parallel(actions: list[tuple[str, int, int, int]], scratch_slot: int
                         ) -> list[MI]:
    """Sequentialise a parallel assignment of registers and memory slots.

    ``actions`` items are ``(kind, dst, src, slot)``:

    * ``("copy", dst_reg, src_reg, -1)``
    * ``("store", -1, src_reg, slot)``      # slot <- src_reg

    Algorithm: repeatedly emit an action whose written register is not still
    needed as a source by another pending action.  Stores write no register
    and are always eligible; loads (generated while breaking cycles) are
    eligible once their destination is no longer needed.  When only register
    permutation cycles remain, one cycle register is saved to
    ``scratch_slot`` and the action consuming it becomes a load.  The result
    never needs a spare register, so it is correct for every K >= 1.
    Deterministic: pending actions keep input order; the first eligible one
    is chosen.
    """
    pending: list[list] = []
    for kind, dst, src, slot in actions:
        if kind == "copy" and dst == src:
            continue  # coalesced away
        pending.append([kind, dst, src, slot])

    out: list[MI] = []

    def emit(e: list) -> None:
        if e[0] == "copy":
            out.append(MI(kind="copy", dst=e[1], src=e[2]))
        elif e[0] == "store":
            out.append(MI(kind="store", src=e[2], slot=e[3]))
        else:  # load
            out.append(MI(kind="load", dst=e[1], slot=e[3]))

    while pending:
        needed_srcs = {e[2] for e in pending if e[0] in ("copy", "store")}
        move = None
        for e in pending:
            if e[0] == "store" or e[1] not in needed_srcs:
                move = e
                break
        if move is not None:
            pending.remove(move)
            emit(move)
            continue

        # only register-to-register permutation cycles remain
        cyc = next(e for e in pending if e[0] == "copy")
        saved = cyc[1]
        tail = next(e for e in pending if e[0] == "copy" and e[2] == saved)
        out.append(MI(kind="store", src=saved, slot=scratch_slot))
        tail[0] = "load"
        tail[2] = -1
        tail[3] = scratch_slot
    return out


def lower_program(program: Program, vp: VirtualProgram, color: dict[int, int],
                  remat_info=None) -> AllocatedProgram:
    n = len(program.blocks)
    blocks = [ABlock(label=b.label, succs=list(b.succs), origin=i) for i, b in enumerate(program.blocks)]
    scratch_slot = program.var_count()

    # ---- instruction bodies (pre actions / op / post actions) -------------
    def emit_materialised(mi_list, actions):
        """Emit load/remat materialisation actions in recorded order."""
        for kind, a, x, *rest in actions:
            if kind == "load":
                mi_list.append(MI(kind="load", dst=color[a], slot=x))
            else:  # remat: x is the def uid, rest[0] are operand nodes
                op_nodes = rest[0]
                mi_list.append(MI(
                    kind="remat", dst=color[a],
                    reads=tuple(color[o] for o in op_nodes), uid=x,
                ))

    for i, b in enumerate(program.blocks):
        for ins in b.instrs:
            plan: InstrPlan = vp.instr_plans[ins.uid]
            for kind, a, v, *rest in plan.pre:
                if kind == "load":
                    blocks[i].instrs.append(MI(kind="load", dst=color[a], slot=v))
                elif kind == "remat":
                    blocks[i].instrs.append(MI(
                        kind="remat", dst=color[a],
                        reads=tuple(color[o] for o in rest[0]), uid=v,
                    ))
                else:  # copy pseudo <- operand
                    if color[a] != color[v]:
                        blocks[i].instrs.append(MI(kind="copy", dst=color[a], src=color[v]))
            if not plan.elided:
                reads = tuple(color[r] for r in plan.reads)
                writes = tuple(color[w] for w in plan.writes)
                blocks[i].instrs.append(
                    MI(kind="op", reads=reads, writes=writes,
                       clobbers=tuple(sorted(set(ins.clobbers))), uid=ins.uid, note=ins.note)
                )
            for kind, a, v in plan.post:
                if kind == "store":
                    blocks[i].instrs.append(MI(kind="store", src=color[a], slot=v))
                else:  # copy dst <- src (pseudo carries the fixed result)
                    if color[a] != color[v]:
                        blocks[i].instrs.append(MI(kind="copy", dst=color[a], src=color[v]))

    # ---- edge actions: append or split -------------------------------------
    def emit_edge_actions(mi_list: list[MI], ep: EdgePlan) -> None:
        emit_materialised(mi_list, ep.loads)
        emit_materialised(mi_list, ep.remats)
        actions: list[tuple[str, int, int, int]] = []
        for kind, d, s in ep.copies:
            rd, rs = color[d], color[s]
            actions.append(("copy", rd, rs, -1))
        for kind, s, dvar in ep.stores:
            actions.append(("store", -1, color[s], dvar))
        mi_list.extend(_sequential_parallel(actions, scratch_slot))

    extra: list[ABlock] = []
    for (p, q), ep in vp.edge_plans.items():
        if not ep.loads and not ep.remats and not ep.copies and not ep.stores:
            continue
        if len(program.blocks[p].succs) == 1:
            emit_edge_actions(blocks[p].instrs, ep)
        else:
            split = ABlock(
                label=f"{blocks[p].label}__split_{blocks[q].label}",
                succs=[q],
                origin=-1,
                split_edge=(p, q),
            )
            sid = n + len(extra)
            extra.append(split)
            blocks[p].succs[blocks[p].succs.index(q)] = sid
            emit_edge_actions(split.instrs, ep)

    all_blocks = tuple(blocks + extra)
    slots = {v: v for v in sorted(vp.spilled)}
    assignment = {v: color[v] for v in vp.real_nodes if v in color}
    return AllocatedProgram(
        blocks=all_blocks,
        k=_k_from_color(color),
        slots=slots,
        assignment=assignment,
        rematerialized={v: remat_info.costs[remat_info.eligible[v]]
                         for v in sorted(vp.rematerialized)} if remat_info else {},
        scratch_slot=scratch_slot,
        param_regs={v: assignment[v] for v in program.params if v in assignment},
    )


def _k_from_color(color: dict[int, int]) -> int:
    return max(color.values()) + 1 if color else 0
