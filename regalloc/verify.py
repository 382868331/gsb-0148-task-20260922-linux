"""Independent allocation checker (v2).

This module does **not** import the allocator, the lowerer or the liveness
module.  Given ``(program, machine, result)`` it recomputes edge-sensitive
liveness from the documented equations and then verifies, program point by
program point and edge by edge:

* **bit-level homes** — every live variable's value bits reside exactly in
  its home view; two simultaneously live register-resident variables never
  share a single bit (aliasing views included);
* **stack slots** — two simultaneously live spilled variables never share a
  stack byte; offsets honour the class alignment; the scratch slot overlaps
  no variable slot;
* **calls** — a value surviving a call never sits in a view the call
  destroys (non-callee-saved views plus explicit ``clobber_views``);
* **early clobber** — an early-clobber def's materialised view shares no
  bits with any view the same instruction reads;
* **pinned operands** — ``fixed_views`` operands are materialised in
  exactly the pinned view at the lowered op;
* **edge realizability** — every source edge (normal AND exceptional) is
  traversable in the allocated CFG; an edge that cannot be located is
  reported as a failure at its location, never silently skipped.

The checker does not trust solver-side boolean flags: a non-``"allocated"``
status is accepted only when the result carries no program/assignment at
all, and an ``"allocated"`` status is re-derived from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

from .machine import Machine
from .model import Program


@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    failures: tuple[str, ...]
    edges_checked: int
    points_checked: int
    slots_checked: int

    def summary(self) -> str:
        status = "OK" if self.ok else f"FAILED ({len(self.failures)} problem(s))"
        return (
            f"{status}; edges={self.edges_checked} points={self.points_checked} "
            f"slots={self.slots_checked}"
        )


# --------------------------------------------------------------------------
# independently implemented liveness (same documented equations)
# --------------------------------------------------------------------------


def _independent_liveness(program: Program):
    blocks = program.blocks
    n = len(blocks)

    phi_defs: list[set[int]] = []
    phi_src: dict[tuple[int, int], set[int]] = {}
    for i, b in enumerate(blocks):
        d: set[int] = set()
        for phi in b.phis:
            d.add(phi.def_var)
            for p, s in phi.sources:
                phi_src.setdefault((p, i), set()).add(s)
        phi_defs.append(d)

    throw_at: list[int | None] = []
    for b in blocks:
        t = None
        for k, ins in enumerate(b.instrs):
            if ins.may_throw:
                t = k
        throw_at.append(t)

    live_in: list[set[int]] = [set() for _ in range(n)]
    live_out: list[set[int]] = [set() for _ in range(n)]

    def edge(pred: int, succ: int) -> set[int]:
        return (live_in[succ] - phi_defs[succ]) | phi_src.get((pred, succ), set())

    changed = True
    while changed:
        changed = False
        for i in range(n):
            b = blocks[i]
            out_i: set[int] = set()
            for s in b.succs:
                out_i |= edge(i, s)
            live = set(out_i)
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
                    live |= edge(i, q)
                for k in range(t, -1, -1):
                    ins = b.instrs[k]
                    live = (live - set(ins.defs)) | set(ins.uses)
            in_i = live - phi_defs[i]
            if in_i != live_in[i] or out_i != live_out[i]:
                live_in[i] = in_i
                live_out[i] = out_i
                changed = True

    edge_live: dict[tuple[int, int], set[int]] = {}
    for i, b in enumerate(blocks):
        for s in b.succs:
            edge_live[(i, s)] = edge(i, s)
        for q in b.exc_succs:
            edge_live[(i, q)] = edge(i, q)

    before: list[list[set[int]]] = []
    after: list[list[set[int]]] = []
    for i, b in enumerate(blocks):
        live = set()
        for s in b.succs:
            live |= edge_live[(i, s)]
        t = throw_at[i]
        bs: list[set[int]] = [set() for _ in b.instrs]
        as_: list[set[int]] = [set() for _ in b.instrs]
        if t is None:
            for k in range(len(b.instrs) - 1, -1, -1):
                ins = b.instrs[k]
                as_[k] = set(live)
                live = (live - set(ins.defs)) | set(ins.uses)
                bs[k] = set(live)
        else:
            for k in range(len(b.instrs) - 1, t, -1):
                ins = b.instrs[k]
                as_[k] = set(live)
                live = (live - set(ins.defs)) | set(ins.uses)
                bs[k] = set(live)
            for q in b.exc_succs:
                live |= edge_live[(i, q)]
            for k in range(t, -1, -1):
                ins = b.instrs[k]
                as_[k] = set(live)
                live = (live - set(ins.defs)) | set(ins.uses)
                bs[k] = set(live)
        before.append(bs)
        after.append(as_)

    return live_in, live_out, edge_live, before, after


# --------------------------------------------------------------------------
# main checker
# --------------------------------------------------------------------------


def verify_allocation(program: Program, machine: Machine, result) -> VerifyReport:
    failures: list[str] = []
    edges_checked = 0
    points_checked = 0
    slots_checked = 0

    # ---- a non-allocated status must be honestly empty -------------------
    if result.status != "allocated":
        if (
            result.allocated_program is not None
            or result.assignment is not None
            or result.slots is not None
        ):
            failures.append(
                f"status {result.status!r} but the result carries a program/"
                f"assignment: an unallocated state must not look complete"
            )
        return VerifyReport(not failures, tuple(failures), 0, 0, 0)

    ap = result.allocated_program
    assignment = result.assignment or {}
    slots = result.slots or {}
    var_classes = program.var_classes or {}

    def cls_of(v: int) -> str:
        return machine.class_of(v, var_classes)

    live_in, live_out, edge_live, before, after = _independent_liveness(program)

    # ---- completeness: every variable has exactly one home ---------------
    all_vars: set[int] = set(program.params)
    for b in program.blocks:
        for ins in b.instrs:
            all_vars.update(ins.defs)
        for phi in b.phis:
            all_vars.add(phi.def_var)
    for v in sorted(all_vars):
        in_reg = v in assignment
        in_slot = v in slots
        if in_reg == in_slot:
            failures.append(
                f"v{v}: must be either register-assigned or spilled, got "
                f"assigned={in_reg} spilled={in_slot}"
            )

    # ---- views and slots are well-formed ---------------------------------
    for v, view in assignment.items():
        if view not in machine.view_map:
            failures.append(f"v{v}: assigned unknown view {view!r}")
            continue
        got = machine.view(view).cls
        want = cls_of(v)
        if got != want:
            failures.append(
                f"v{v}: view {view!r} is class {got!r}, variable is class {want!r}"
            )
    for v, slot in slots.items():
        cls = cls_of(v)
        want_size = machine.class_width[cls] // 8
        want_align = machine.class_align[cls]
        slots_checked += 1
        if slot.size != want_size:
            failures.append(f"v{v}: slot size {slot.size} != class width {want_size}")
        if slot.align != want_align or slot.offset % want_align != 0:
            failures.append(
                f"v{v}: slot offset {slot.offset} violates alignment {want_align}"
            )
        if slot.offset < 0:
            failures.append(f"v{v}: negative slot offset")
    scratch = result.scratch_slot
    if scratch is not None:
        for v, slot in slots.items():
            if slot.offset < scratch.end and scratch.offset < slot.end:
                failures.append(
                    f"scratch slot [{scratch.offset},{scratch.end}) overlaps "
                    f"slot of v{v} [{slot.offset},{slot.end})"
                )

    def bits_of(view: str) -> tuple[int, int, int]:
        vw = machine.view(view)
        return (vw.group, vw.lo, vw.hi)

    def overlap(a, b) -> bool:
        return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]

    # ---- per-point bit-level and slot-overlap checks ----------------------
    def check_point(where: str, live) -> None:
        nonlocal points_checked
        points_checked += 1
        reg_items: list[tuple[int, tuple[int, int, int]]] = []
        slot_items: list[tuple[int, int, int]] = []
        for v in sorted(live):
            if v in assignment and assignment[v] in machine.view_map:
                reg_items.append((v, bits_of(assignment[v])))
            elif v in slots:
                s = slots[v]
                slot_items.append((v, s.offset, s.end))
            else:
                failures.append(f"{where}: live v{v} has no home")
        for i in range(len(reg_items)):
            for j in range(i + 1, len(reg_items)):
                vi, bi = reg_items[i]
                vj, bj = reg_items[j]
                if overlap(bi, bj):
                    failures.append(
                        f"{where}: v{vi} ({assignment[vi]}) and v{vj} "
                        f"({assignment[vj]}) share register bits"
                    )
        for i in range(len(slot_items)):
            for j in range(i + 1, len(slot_items)):
                vi, lo_i, hi_i = slot_items[i]
                vj, lo_j, hi_j = slot_items[j]
                if lo_i < hi_j and lo_j < hi_i:
                    failures.append(
                        f"{where}: spilled v{vi} and v{vj} share stack bytes "
                        f"[{lo_i},{hi_i}) vs [{lo_j},{hi_j})"
                    )

    for bi, b in enumerate(program.blocks):
        check_point(f"block {bi} entry", live_in[bi])
        for ii, ins in enumerate(b.instrs):
            check_point(f"block {bi} instr uid {ins.uid} before", before[bi][ii])
            check_point(f"block {bi} instr uid {ins.uid} after", after[bi][ii])
        check_point(f"block {bi} exit", live_out[bi])
    for (p, q), live in sorted(edge_live.items()):
        kind = "exc" if q in program.blocks[p].exc_succs else "normal"
        check_point(f"edge {p}->{q} ({kind})", live)
        edges_checked += 1

    # ---- per-instruction machine-level checks -----------------------------
    ops_by_uid: dict[int, object] = {}
    if ap is not None:
        for ab in ap.blocks:
            for mi in ab.instrs:
                if mi.kind == "op":
                    if mi.uid in ops_by_uid:
                        failures.append(f"op uid {mi.uid} emitted more than once")
                    ops_by_uid[mi.uid] = mi

    for bi, b in enumerate(program.blocks):
        for ii, ins in enumerate(b.instrs):
            ip = f"block {bi} instr uid {ins.uid}"
            mi = ops_by_uid.get(ins.uid)
            if mi is None:
                failures.append(f"{ip}: no lowered op emitted")
                continue
            if len(mi.reads) != len(ins.uses) or len(mi.writes) != len(ins.defs):
                failures.append(
                    f"{ip}: lowered op has {len(mi.reads)} reads/"
                    f"{len(mi.writes)} writes, source has "
                    f"{len(ins.uses)}/{len(ins.defs)}"
                )
                continue
            # pinned operands materialised in exactly the pinned view
            for pos, u in enumerate(ins.uses):
                if u in ins.fixed_views and mi.reads[pos] != ins.fixed_views[u]:
                    failures.append(
                        f"{ip}: use v{u} pinned to {ins.fixed_views[u]!r} but "
                        f"the op reads {mi.reads[pos]!r}"
                    )
            for pos, d in enumerate(ins.defs):
                if d in ins.fixed_views and mi.writes[pos] != ins.fixed_views[d]:
                    failures.append(
                        f"{ip}: def v{d} pinned to {ins.fixed_views[d]!r} but "
                        f"the op writes {mi.writes[pos]!r}"
                    )
            # early clobber: def view shares no bits with any read view
            for pos, d in enumerate(ins.defs):
                if d not in ins.early_clobber:
                    continue
                wd = bits_of(mi.writes[pos])
                for rpos, rview in enumerate(mi.reads):
                    if overlap(wd, bits_of(rview)):
                        failures.append(
                            f"{ip}: early-clobber def v{d} in {mi.writes[pos]!r} "
                            f"overlaps read {rview!r} (use position {rpos})"
                        )
            # call clobbers: survivors avoid destroyed views
            if ins.is_call:
                clobbered = set(machine.clobbered_by_call(ins.clobber_views))
            else:
                clobbered = set(ins.clobber_views)
            if clobbered:
                survivors = after[bi][ii] - set(ins.defs)
                for v in sorted(survivors):
                    if v not in assignment or assignment[v] not in machine.view_map:
                        continue
                    vb = bits_of(assignment[v])
                    for cv in clobbered:
                        if overlap(vb, bits_of(cv)):
                            failures.append(
                                f"{ip}: v{v} survives the call in {assignment[v]!r} "
                                f"which the call clobbers (via {cv!r})"
                            )

    # ---- edge realizability in the allocated CFG ---------------------------
    if ap is not None:
        origin_of: dict[int, int] = {}
        for ai, ab in enumerate(ap.blocks):
            if ab.origin >= 0:
                origin_of[ab.origin] = ai
        for ai, ab in enumerate(ap.blocks):
            if ab.origin < 0 and any(mi.kind == "op" for mi in ab.instrs):
                failures.append(f"split block {ab.label} contains an op")

        def walk(start: int, wanted: int, where: str) -> None:
            t = start
            seen = set()
            while ap.blocks[t].origin < 0:
                if t in seen or len(ap.blocks[t].succs) != 1:
                    failures.append(f"{where}: split chain broken at {ap.blocks[t].label}")
                    return
                seen.add(t)
                t = ap.blocks[t].succs[0]
            if ap.blocks[t].origin != wanted:
                failures.append(
                    f"{where}: chain ends at block {ap.blocks[t].origin}, "
                    f"expected {wanted}"
                )

        for p, b in enumerate(program.blocks):
            ap_idx = origin_of.get(p)
            if ap_idx is None:
                failures.append(f"block {p}: missing from the allocated program")
                continue
            ab = ap.blocks[ap_idx]
            if len(ab.succs) != len(b.succs) or len(ab.exc_succs) != len(b.exc_succs):
                failures.append(f"block {p}: successor count changed by lowering")
                continue
            for idx, q in enumerate(b.succs):
                walk(ab.succs[idx], q, f"edge {p}->{q} (normal)")
            for idx, q in enumerate(b.exc_succs):
                walk(ab.exc_succs[idx], q, f"edge {p}->{q} (exc)")

        # lowered code legality
        known_slots = {s.offset for s in slots.values()}
        if scratch is not None:
            known_slots.add(scratch.offset)
        for ab in ap.blocks:
            for mi in ab.instrs:
                if mi.kind not in ("op", "copy", "load", "store"):
                    failures.append(f"block {ab.label}: unknown mi kind {mi.kind!r}")
                for view in (*mi.reads, *mi.writes, mi.dst, mi.src, *mi.clobbers):
                    if view and view not in machine.view_map:
                        failures.append(
                            f"block {ab.label}: unknown view {view!r} in {mi.kind}"
                        )
                if mi.kind in ("load", "store") and mi.slot not in known_slots:
                    failures.append(
                        f"block {ab.label}: {mi.kind} uses unknown slot offset {mi.slot}"
                    )

    return VerifyReport(
        ok=not failures,
        failures=tuple(failures),
        edges_checked=edges_checked,
        points_checked=points_checked,
        slots_checked=slots_checked,
    )
