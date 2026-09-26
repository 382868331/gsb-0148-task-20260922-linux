"""Iteration demo: stack-slot layout with lifetime-based slot sharing.

Run with::

    python demo_iteration.py

Everything is computed on the spot (no sleep, no pre-recorded output):

1. the high-pressure straight-line program is allocated twice - with the
   packing switch off and on - showing the independent per-spill frame vs
   the packed minimum frame, with every load/store final offset;
2. the branch-mutex/join contrast: branch-private spilled values share one
   offset across mutually exclusive edges while values crossing the join
   stay in disjoint byte ranges;
3. a value spilled across a call survives it in memory;
4. the size/alignment knobs and the budget boundary (budget exhausted ->
   legal layout, explicitly not proven optimal);
5. the independent paired symbolic executor verifies the real memory
   accesses of every packed result.
"""

from __future__ import annotations

import time

from regalloc import allocate
from regalloc.samples import (
    sample_branch_slot_reuse,
    sample_register_pressure,
    sample_spilled_across_call,
)
from regalloc.symbolic import equivalence_check


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _check(prog, alloc, label: str) -> None:
    res = equivalence_check(prog, alloc)
    print(f"  paired symbolic execution ({label}): {res.summary()}")
    assert res.equivalent, res.failure


def show_pressure() -> None:
    hr("1) STRAIGHT-LINE PRESSURE - independent slots vs packed frame")
    prog = sample_register_pressure()
    plain = allocate(prog, 2)
    packed = allocate(prog, 2, pack_slots=True)
    n_logical = len(packed.stack_layout.slot_offsets)
    independent_bytes = 8 * n_logical
    print(f"  logical slots (spilled vars + scratch): {n_logical}")
    print(f"  frame with one independent slot each: {independent_bytes} bytes")
    print(f"  packed frame size: {packed.stack_layout.frame_size} bytes "
          f"(alignment {packed.stack_layout.frame_alignment})")
    print(f"  optimal: {packed.stack_layout.proven_optimal} "
          f"({packed.stack_layout.note})")
    print("  register assignment identical with switch off/on:",
          plain.assignment == packed.assignment and plain.slots == packed.slots)
    print("  logical slot -> final byte offset:")
    for slot, off in packed.stack_layout.slot_offsets.items():
        kind = "scratch" if slot == packed.allocated_program.scratch_slot else f"v{slot}"
        print(f"    {kind:>8}: fp{off:+d}")
    _check(prog, packed, "packed K=2")


def show_branch_contrast() -> None:
    hr("2) BRANCH-MUTEX SLOTS REUSE; JOIN-CROSSING VALUES CANNOT")
    prog = sample_branch_slot_reuse()
    a = allocate(prog, 2, pack_slots=True)
    off = a.stack_layout.slot_offsets
    print("  spilled variables:", sorted(a.slots))
    print("  slot -> offset:", {s: off[s] for s in sorted(a.slots)})
    priv_l = sorted(v for v in (3, 4) if v in a.slots)
    priv_r = sorted(v for v in (7, 8) if v in a.slots)
    for v in priv_l + priv_r:
        arm = "left " if v in priv_l else "right"
        peers = [w for w in (priv_l if v in priv_r else priv_r) if off[w] == off[v]]
        later = sorted(w for w in a.slots if w > v and off[w] == off[v])
        print(f"  {arm}-private v{v} at fp{off[v]:+d}; same offset also used by "
              f"{[f'v{w}' for w in later] or '(earliest occupant)'}")
    cross = sorted(v for v in (1, 2) if v in a.slots)
    print("  values crossing the join (distinct offsets, cannot share with "
          "overlapping arm values):",
          [f"v{v}@fp{off[v]:+d}" for v in cross])
    print(f"  frame size: {a.stack_layout.frame_size}; optimal proven: "
          f"{a.stack_layout.proven_optimal}")
    _check(prog, a, "both branches, phi edges")


def show_call() -> None:
    hr("3) SPILLED VALUE LIVE ACROSS A CALL (kept in memory, not lost)")
    prog = sample_spilled_across_call()
    a = allocate(prog, 2, pack_slots=True)
    print("  spilled variables:", sorted(a.slots))
    print("  frame size:", a.stack_layout.frame_size, "bytes;",
          "proven optimal:", a.stack_layout.proven_optimal)
    print("  lowered code (offsets on every load/store):")
    for line in a.allocated_program.render().splitlines():
        print("   ", line)
    _check(prog, a, "call clobber r0,r1")


def show_knobs_and_boundary() -> None:
    hr("4) CALLER-GIVEN SIZES/ALIGNMENTS AND THE BUDGET BOUNDARY")
    prog = sample_register_pressure()
    wide = allocate(
        prog, 2, pack_slots=True, slot_size=16, slot_alignment=16,
        frame_alignment=32,
    )
    print("  slot_size=16, slot_alignment=16, frame_alignment=32:")
    print("    frame:", wide.stack_layout.frame_size, "bytes;",
          "proven optimal:", wide.stack_layout.proven_optimal)
    print("    offsets:", wide.stack_layout.slot_offsets)
    _check(prog, wide, "wide slots")

    tight = allocate(prog, 2, pack_slots=True, layout_budget=1)
    print("  layout_budget=1:")
    print("    proven optimal:", tight.stack_layout.proven_optimal)
    print("    frame:", tight.stack_layout.frame_size, "bytes (legal layout)")
    print("    note:", tight.stack_layout.note)
    _check(prog, tight, "budget-exhausted legal layout")


def main() -> None:
    t0 = time.perf_counter()
    show_pressure()
    show_branch_contrast()
    show_call()
    show_knobs_and_boundary()
    hr("ITERATION DEMO COMPLETE")
    print(f"total wall time: {time.perf_counter() - t0:.2f} s (budget: 90 s)")


if __name__ == "__main__":
    main()
