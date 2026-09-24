"""Lowering for format v2.

Turns a bank-set coloring + spill plan into register-view-level machine
code.  Machine instructions (:class:`MI2`) name concrete *views*:

    op     reads=(view,..) writes=(view,..) clobbersv=(view,..)
    copy   dst_view <- src_view
    load   dst_view <- slot[start + off]
    store  slot[start + off] <- src_view

CFG construction
================
A source block containing excepting instructions is cut into segments:
segment 0 ends just before the first excepting instruction; the segment that
*contains* an excepting instruction has one successor per exception target
(each running that edge's parallel copies in a dedicated split block) plus
its normal continuation.  Normal edge copies are appended to the final
segment only when that edge is non-critical and the block has no exception
transfers; otherwise they live in a split block too.

Parallel copies execute simultaneously over *values* (a double-width value
is one atomic action touching two views).  Remaining permutation cycles are
broken, in order of preference, with

1. a genuinely free register-view option of the value's class whose banks
   alias no source, destination or crossing-live location at that point; or
2. a private, aligned scratch stack slot beyond the data frame (so it can
   never overlap a live value's slot).

A still-live aliased register view is therefore never borrowed, and when no
free temporary register exists (the small target requirement) the cycle
goes through memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Program, RegAllocError


@dataclass(frozen=True)
class MI2:
    kind: str                    # "op" | "copy" | "load" | "store"
    dst: str = ""                # copy/load destination view
    src: str = ""                # copy source view
    slot: int = -1               # frame start
    off: int = 0                 # unit offset inside the slot
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    clobbersv: tuple[str, ...] = ()
    uid: int = -1
    note: str = ""

    def render(self) -> str:
        if self.kind == "op":
            rd = ",".join(self.reads)
            wr = ",".join(self.writes)
            cl = (f" clobber[{','.join(self.clobbersv)}]" if self.clobbersv else "")
            tag = f" ; {self.note}" if self.note else ""
            return f"op{self.uid:<3d} {wr} = f({rd}){cl}{tag}"
        if self.kind == "copy":
            return f"copy {self.dst} <- {self.src}"
        if self.kind == "load":
            return f"load {self.dst} <- slot[{self.slot + self.off}]"
        return f"store slot[{self.slot + self.off}] <- {self.src}"


@dataclass
class ABlock2:
    label: str
    instrs: list[MI2] = field(default_factory=list)
    succs: list[int] = field(default_factory=list)
    origin: int = -1
    seg: int = 0                 # segment index within the origin block
    split_edge: tuple[int, int] | None = None
    exc_instr: int | None = None  # exception split: excepting instr index
    # origin block holding an excepting instruction's NORMAL-path post
    # actions (the result is not written on an exception transfer)
    normal_post_of: int | None = None
    normal_post_instr: int = -1


@dataclass
class AllocatedProgram2:
    blocks: tuple[ABlock2, ...]
    target_name: str
    num_banks: int
    slots: dict[int, tuple[int, int]]
    assignment: dict[int, tuple[str, ...]]
    frame_slots: int
    scratch_slot: int
    scratch_size: int
    param_views: dict[int, tuple[str, ...]]

    def render(self) -> str:
        lines: list[str] = []
        for i, b in enumerate(self.blocks):
            tail = (" -> " + ",".join(self.blocks[s].label for s in b.succs)) if b.succs else ""
            tag = ""
            if b.split_edge is not None:
                kind = "exc" if b.exc_instr is not None else "edge"
                tag = f"  ; {kind} split of {b.split_edge[0]}->{b.split_edge[1]}"
            lines.append(f"{b.label}:{tail}{tag}")
            for mi in b.instrs:
                lines.append(f"    {mi.render()}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# value-level parallel assignment sequentialisation
# --------------------------------------------------------------------------

@dataclass
class _VAction:
    kind: str                    # "copy" | "store" | "load"
    cls: str = ""
    dst_views: tuple[str, ...] = ()
    src_views: tuple[str, ...] = ()
    slot: int = -1
    size: int = 1


def _banks_of(plan, color, node):
    if node in plan.precolor:
        return plan.precolor[node]
    return color[node]


def _build_actions(rt, plan, color, records) -> list[_VAction]:
    """Resolve node-level records into concrete view-level value actions."""
    actions: list[_VAction] = []
    for rec in records:
        kind = rec[0]
        if kind == "copy":
            _, d, s = rec
            dv = rt.home_view(plan.node_class[d], _banks_of(plan, color, d))
            sv = rt.home_view(plan.node_class[s], _banks_of(plan, color, s))
            if dv == sv:
                continue
            actions.append(_VAction(
                "copy", cls=plan.node_class[d], dst_views=dv, src_views=sv,
                size=max(len(dv), len(sv)),
            ))
        elif kind == "load":
            _, d, start = rec
            dv = rt.home_view(plan.node_class[d], _banks_of(plan, color, d))
            actions.append(_VAction(
                "load", cls=plan.node_class[d], dst_views=dv,
                slot=start, size=len(dv),
            ))
        else:  # store
            _, s, start = rec
            sv = rt.home_view(plan.node_class[s], _banks_of(plan, color, s))
            actions.append(_VAction(
                "store", cls=plan.node_class[s], src_views=sv,
                slot=start, size=len(sv),
            ))
    return actions


def emit_parallel(rt, plan, color, records, *, crossing_banks,
                  scratch_slot, scratch_size) -> list[MI2]:
    """Sequentialise one edge's parallel assignment.

    Precondition satisfied by the caller: ``records`` already carry concrete
    spill-slot starts (not spilled variable ids).
    """
    actions = _build_actions(rt, plan, color, records)
    out: list[MI2] = []
    pending = list(actions)

    def bank(name):
        return rt.view(name).bank

    while pending:
        # registers still needed as a source by a pending action
        needed = {bank(v) for a in pending for v in a.src_views}
        move = None
        for a in pending:
            if a.kind in ("store", "load"):
                move = a
                break
            dst_banks = {bank(v) for v in a.dst_views}
            if not (dst_banks & needed):
                move = a
                break
        if move is not None:
            pending.remove(move)
            if move.kind == "copy":
                for d, s in zip(move.dst_views, move.src_views):
                    out.append(MI2(kind="copy", dst=d, src=s))
            elif move.kind == "load":
                for k, vn in enumerate(move.dst_views):
                    out.append(MI2(kind="load", dst=vn, slot=move.slot, off=k))
            else:
                for k, vn in enumerate(move.src_views):
                    out.append(MI2(kind="store", src=vn, slot=move.slot, off=k))
            continue

        # only register-to-register permutation cycles remain.
        cyc = next(a for a in pending if a.kind == "copy")

        # (1) genuinely free register option: its banks alias no pending
        # source, destination, or crossing-live location.
        dst_banks_cycle = {
            bank(v) for a in pending if a.kind == "copy" for v in a.dst_views
        }
        blocked = frozenset(set(needed) | dst_banks_cycle | set(crossing_banks))
        free_views = rt.free_temp_view(cyc.cls, blocked)
        saved_views = cyc.dst_views
        if free_views is not None:
            for tmp, sv in zip(free_views, saved_views):
                out.append(MI2(kind="copy", dst=tmp, src=sv))
            tail = next(
                a for a in pending
                if a.kind == "copy"
                and {bank(v) for v in a.src_views}
                == {bank(v) for v in saved_views}
            )
            tail.src_views = free_views
            continue

        # (2) no free temporary register: private scratch stack slot.
        if scratch_size < cyc.size:  # pragma: no cover - sized by caller
            raise RegAllocError("internal: scratch slot too small for cycle")
        for k, vn in enumerate(saved_views):
            out.append(MI2(kind="store", src=vn, slot=scratch_slot, off=k))
        tail = next(
            a for a in pending
            if a.kind == "copy"
            and {bank(v) for v in a.src_views}
            == {bank(v) for v in saved_views}
        )
        tail.kind = "load"
        tail.slot = scratch_slot
        tail.src_views = ()
    return out


# --------------------------------------------------------------------------
# main lowering
# --------------------------------------------------------------------------

def _align_up(value: int, align: int) -> int:
    if align > 1 and value % align:
        return value + align - (value % align)
    return value


def lower_v2(program: Program, plan, color, placement, frame):
    rt = program.target.resolved()

    # scratch frame: big enough for the widest value that can head a cycle
    scratch_size = 1
    for records in plan.edge_records.values():
        for rec in records:
            if rec[0] == "copy":
                node = rec[1]
                if rt.class_info(plan.node_class[node]).is_pair:
                    scratch_size = 2
    align = program.stack.align
    scratch_slot = _align_up(frame, align)
    frame_final = _align_up(scratch_slot + scratch_size, align)

    def slot_start(var: int) -> int:
        return placement[var][0]

    def home_views(node: int) -> tuple[str, ...]:
        return rt.home_view(plan.node_class[node], _banks_of(plan, color, node))

    def expand_actions(items: list[tuple], out: list[MI2]) -> None:
        for kind, a, v in items:
            if kind == "copy":
                for x, y in zip(home_views(a), home_views(v)):
                    if x != y:
                        out.append(MI2(kind="copy", dst=x, src=y))
            elif kind == "load":
                start = slot_start(v)
                for k, vn in enumerate(home_views(a)):
                    out.append(MI2(kind="load", dst=vn, slot=start, off=k))
            else:  # store
                start = slot_start(v)
                for k, vn in enumerate(home_views(a)):
                    out.append(MI2(kind="store", src=vn, slot=start, off=k))

    # ---- segment the original blocks --------------------------------------
    ablocks: list[ABlock2] = []
    seg_of_instr: list[list[int]] = []
    first_seg: list[int] = []
    # per source block: list of (ablock sid, ends_with_exc_instr_index|None)
    seg_meta: list[list[tuple[int, int | None]]] = []

    for i, b in enumerate(program.blocks):
        exc_points = [j for j, ins in enumerate(b.instrs) if ins.except_to]
        # segment k ends WITH exc_points[k] inclusive; a final tail follows
        starts = [0] + [j + 1 for j in exc_points]
        ends = [j + 1 for j in exc_points] + [len(b.instrs)]
        multi = len(starts) > 1
        per_instr = [-1] * len(b.instrs)
        meta: list[tuple[int, int | None]] = []
        for k, (st, en) in enumerate(zip(starts, ends)):
            if st == en:
                continue  # empty tail segment (last instr was excepting)
            ab = ABlock2(
                label=f"{b.label}__s{k}" if multi else b.label,
                origin=i, seg=k,
            )
            ablocks.append(ab)
            sid = len(ablocks) - 1
            for j in range(st, en):
                per_instr[j] = sid
            exc_j = exc_points[k] if k < len(exc_points) else None
            meta.append((sid, exc_j))
        # a block that produced no non-empty segment (e.g. a block with zero
        # instructions, or whose only instruction excepts and its tail ran
        # together) still needs one origin block for successor wiring
        if not meta:
            ab = ABlock2(label=b.label, origin=i, seg=0)
            ablocks.append(ab)
            meta.append((len(ablocks) - 1, None))
        seg_of_instr.append(per_instr)
        first_seg.append(meta[0][0])
        seg_meta.append(meta)

    # ---- emit instruction bodies ------------------------------------------
    # A segment ends with the excepting instruction.  Its post-actions may
    # store a result that is never written on the exception path (and the
    # slot could be reused there), so they are emitted on the NORMAL
    # continuation only, collected in deferred_post[(block, instr_index)].
    deferred_post: dict[tuple[int, int], list[MI2]] = {}
    for i, b in enumerate(program.blocks):
        exc_set = {j for j, ins in enumerate(b.instrs) if ins.except_to}
        for j, ins in enumerate(b.instrs):
            sid = seg_of_instr[i][j]
            ab = ablocks[sid]
            expand_actions(plan.instr_pre[ins.uid], ab.instrs)
            read_views: list[str] = []
            for node in plan.instr_reads[ins.uid]:
                read_views.extend(home_views(node))
            write_views: list[str] = []
            for node in plan.instr_writes[ins.uid]:
                write_views.extend(home_views(node))
            ab.instrs.append(MI2(
                kind="op", reads=tuple(read_views), writes=tuple(write_views),
                clobbersv=tuple(sorted(set(ins.clobbersv))),
                uid=ins.uid, note=ins.note,
            ))
            post_mis: list[MI2] = []
            expand_actions(plan.instr_post[ins.uid], post_mis)
            if j in exc_set:
                deferred_post[(i, j)] = post_mis
            else:
                ab.instrs.extend(post_mis)

    # ---- edges -------------------------------------------------------------
    scratch_used: dict[tuple[int, int], tuple[int, int, str]] = {}
    report = plan.report
    extra_splits: list[ABlock2] = []

    def concrete_records(p, s):
        out_recs = []
        for rec in plan.edge_records.get((p, s), ()):
            if rec[0] == "copy":
                out_recs.append(rec)
            elif rec[0] == "load":
                out_recs.append(("load", rec[1], slot_start(rec[2])))
            else:
                out_recs.append(("store", rec[1], slot_start(rec[2])))
        return out_recs

    def emit_edge(p, s, ab) -> bool:
        records = concrete_records(p, s)
        if not records:
            return False
        dst_nodes = {r[1] for r in records if r[0] in ("copy", "load")}
        # same crossing definition as the allocator: successor live-in minus
        # the values born on this edge
        born_vars = {c.dst for c in plan.edge_copies[(p, s)]}
        # dst_nodes are node ids; born variables are the copy destinations.
        crossing_banks: set[int] = set()
        for v in report.live_in[s]:
            if v in plan.spilled or v in born_vars:
                continue
            crossing_banks |= set(color[v])
        before = len(ab.instrs)
        ab.instrs.extend(emit_parallel(
            rt, plan, color, records,
            crossing_banks=frozenset(crossing_banks),
            scratch_slot=scratch_slot, scratch_size=scratch_size,
        ))
        used_scratch = any(
            mi.slot == scratch_slot for mi in ab.instrs[before:]
        )
        if used_scratch:
            scratch_used[(p, s)] = (
                scratch_slot, scratch_size, "stack (no free register view)"
            )
        return True

    def new_split(p, s, *, exc: bool, j=None) -> ABlock2:
        sp = ABlock2(
            label=(f"{program.blocks[p].label}__exc{j}_{program.blocks[s].label}"
                   if exc else f"{program.blocks[p].label}__edge_{program.blocks[s].label}"),
            origin=-1, split_edge=(p, s), exc_instr=j,
        )
        extra_splits.append(sp)
        return sp

    for i, b in enumerate(program.blocks):
        instrs = b.instrs
        exc_targets = {t for ins in instrs for t in ins.except_to}
        meta = seg_meta[i]

        def _wire_normal_successors(from_ab, post_mis=()):
            """Branch ``from_ab`` along the block's normal successors.

            Any excepting-instruction normal-only post actions go into their
            own block first (``post_block``), so exception splits never run
            them.  Returns the block normal control leaves from (the post
            block when one was needed).
            """
            cur = from_ab
            if post_mis:
                np = ABlock2(
                    label=f"{b.label}__normalpost",
                    origin=-1, normal_post_of=i,
                )
                extra_splits.append(np)
                np_idx = len(ablocks) + len(extra_splits) - 1
                np.instrs.extend(post_mis)
                cur.succs.append(np_idx)
                cur = np
            normal_targets = [s for s in b.succs if s not in exc_targets]
            for s in normal_targets:
                sp = new_split(i, s, exc=False)
                sp_idx = len(ablocks) + len(extra_splits) - 1
                cur.succs.append(sp_idx)
                if emit_edge(i, s, sp):
                    sp.succs.append(first_seg[s])
                else:
                    cur.succs.pop()
                    extra_splits.pop()
                    cur.succs.append(first_seg[s])
            return cur

        for k, (sid, exc_j) in enumerate(meta):
            ab = ablocks[sid]
            if exc_j is not None:
                ins = instrs[exc_j]
                post_mis = deferred_post.get((i, exc_j), [])
                for t in ins.except_to:
                    sp = new_split(i, t, exc=True, j=exc_j)
                    sp_idx = len(ablocks) + len(extra_splits) - 1
                    ab.succs.append(sp_idx)
                    emit_edge(i, t, sp)
                    sp.succs.append(first_seg[t])
                # normal continuation (post actions run on this path only)
                if k + 1 < len(meta):
                    if post_mis:
                        np = ABlock2(
                            label=f"{b.label}__normalpost{exc_j}",
                            origin=-1, normal_post_of=i,
                            normal_post_instr=exc_j,
                        )
                        np.instrs.extend(post_mis)
                        extra_splits.append(np)
                        np_idx = len(ablocks) + len(extra_splits) - 1
                        ab.succs.append(np_idx)
                        np.succs.append(meta[k + 1][0])
                    else:
                        ab.succs.append(meta[k + 1][0])
                else:
                    _wire_normal_successors(ab, post_mis)
            else:
                _wire_normal_successors(ab)

    all_blocks = tuple(ablocks + extra_splits)
    slots = {v: placement[v] for v in sorted(plan.spilled)}
    assignment = {v: home_views(v) for v in plan.real_nodes}
    param_views = {v: assignment[v] for v in program.params if v in assignment}
    return AllocatedProgram2(
        blocks=all_blocks,
        target_name=rt.desc.name,
        num_banks=rt.num_banks,
        slots=slots,
        assignment=assignment,
        frame_slots=frame_final,
        scratch_slot=scratch_slot,
        scratch_size=scratch_size,
        param_views=param_views,
    ), scratch_used, frame_final
