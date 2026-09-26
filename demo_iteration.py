"""Offline demonstration for post-allocation stack-slot packing.

Run::

    python demo_iteration.py

The program performs allocation, post-spill edge-sensitive memory liveness,
bounded exact frame-size enumeration and paired symbolic execution.  No
result is cached or pre-rendered.
"""

from __future__ import annotations

import time

from regalloc import allocate
from regalloc.samples import sample_branch_stack_reuse, sample_register_pressure
from regalloc.symbolic import equivalence_check


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_branch_reuse() -> None:
    hr("1) NORMAL: branch-exclusive spills share one stack interval")
    prog = sample_branch_stack_reuse()
    t0 = time.perf_counter()
    alloc = allocate(prog, 2, pack_stack=True, slot_size=8, slot_alignment=8)
    layout = alloc.stack_layout
    elapsed = (time.perf_counter() - t0) * 1000

    print("program:", prog.name)
    print("register assignment:", alloc.assignment)
    print("logical spill slots:", alloc.slots)
    print("final offsets:", layout.offsets)
    print(f"frame_size={layout.frame_size} frame_alignment={layout.frame_alignment}")
    print(f"exact_optimum={layout.optimal} placements_checked={layout.placements_checked}")
    print(f"layout computed in {elapsed:.2f} ms")
    print("load/store offsets:")
    for access in layout.accesses:
        edge = f" edge{access.edge}" if access.edge is not None else " body"
        print(
            f"  block{access.block}[{access.index}] {access.kind:5s} "
            f"v{access.variable} r{access.register} -> sp+{access.offset} ({edge})"
        )

    result = equivalence_check(prog, alloc)
    print("paired execution:", result.summary())
    assert result.equivalent, result.failure
    assert layout.frame_size == 16


def show_pressure_and_budget_boundary() -> None:
    hr("2) BOUNDARY: over-budget search still returns a legal layout")
    prog = sample_register_pressure()
    optimal = allocate(prog, 2, pack_stack=True, slot_size=8, stack_budget=10_000)
    bounded = allocate(prog, 2, pack_stack=True, slot_size=8, stack_budget=0)

    print("optimal high-pressure layout:", optimal.stack_layout.frame_size, "bytes")
    print("  optimal:", optimal.stack_layout.optimal)
    print("zero-budget legal layout:", bounded.stack_layout.frame_size, "bytes")
    print("  optimal:", bounded.stack_layout.optimal)
    print("  budget_exhausted:", bounded.stack_layout.budget_exhausted)

    for a, b in bounded.stack_layout.interferences:
        off_a = bounded.stack_layout.offsets[a]
        off_b = bounded.stack_layout.offsets[b]
        assert not (off_a < off_b + 8 and off_b < off_a + 8)
    result = equivalence_check(prog, bounded)
    print("paired execution of bounded result:", result.summary())
    assert result.equivalent, result.failure


def main() -> None:
    t0 = time.perf_counter()
    show_branch_reuse()
    show_pressure_and_budget_boundary()
    hr("DEMO COMPLETE")
    print(f"total wall time: {time.perf_counter() - t0:.3f} s")


if __name__ == "__main__":
    main()
