"""Offline demonstration: register allocation with call-clobber constraints.

Run with::

    python demo.py

It performs real allocations on deterministic in-repo programs (nothing is
precomputed or fetched).  It prints:

1. a normal allocation that genuinely spills at K=2, with the register/stack
   map, independently verifiable edge-live sets, the lowered machine code and
   the paired symbolic-execution equivalence result;
2. a call-clobber case (a value surviving a call);
3. an actually triggered rejection boundary (provably infeasible fixed-
   register collision), plus a strict-input rejection (bool as K).
"""

from __future__ import annotations

import time

from regalloc import (
    InfeasibleProgramError,
    ProgramFormatError,
    allocate,
)
from regalloc.liveness import liveness_with_phis
from regalloc.samples import (
    sample_call_surviving,
    sample_fixed_collision,
    sample_remat_pressure,
    sample_register_pressure,
)
from regalloc.symbolic import equivalence_check


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_normal_spill() -> None:
    hr("1) NORMAL RESULT — high register pressure, K=2 (spill is mandatory)")
    prog = sample_register_pressure()
    print("program:", prog.name, "| blocks:", len(prog.blocks),
          "| params:", list(prog.params))
    print("instructions:")
    for b in prog.blocks:
        for ins in b.instrs:
            print(f"  uid{ins.uid}: use={list(ins.uses)} def={list(ins.defs)} {ins.note}")

    t0 = time.perf_counter()
    alloc = allocate(prog, 2)
    elapsed = time.perf_counter() - t0

    print(f"\nallocated for K={alloc.k} in {elapsed*1000:.1f} ms; "
          f"iterations={alloc.iterations}")
    print("home registers (v -> r):",
            {f"v{v}": f"r{r}" for v, r in sorted(alloc.assignment.items())})
    print("spill slots (v -> slot), indexed by variable id:",
            {f"v{v}": f"slot{s}" for v, s in sorted(alloc.slots.items())})

    rep = liveness_with_phis(prog)
    print("\nindependently verifiable liveness (hand-checkable):")
    print("  block0 live-in :", sorted(rep.live_in[0]))
    print("  per-instruction live-before:")
    for ins, before in zip(prog.blocks[0].instrs, rep.instr_before[0]):
        print(f"    uid{ins.uid}: {sorted(before)}")

    print("\nlowered machine code:")
    print(alloc.allocated_program.render())

    res = equivalence_check(prog, alloc)
    print("\nsymbolic before/after equivalence:", res.summary())
    assert res.equivalent, res.failure


def show_call_clobber() -> None:
    hr("2) VALUE LIVE ACROSS A CALL that clobbers r0,r1")
    prog = sample_call_surviving()
    for k in (2, 3):
        alloc = allocate(prog, k)
        res = equivalence_check(prog, alloc)
        print(f"K={k}: slots={sorted(alloc.slots)} -> {res.summary()}")
    print("\nK=3 lowered code (no spill; v1 kept out of r0/r1 across the call):")
    print(allocate(prog, 3).allocated_program.render())


def show_rematerialisation() -> None:
    hr("2b) REMATERIALISATION — cheap pure constant recomputed at its use")
    prog = sample_remat_pressure()
    print("program:", prog.name)
    print("uid0 (v1) is a pure constant used only at uid8; caller marks it")
    print("rematerialisable with remat={0: <cost>} (cost is asserted purity).")

    plain = allocate(prog, 2)
    marked = allocate(prog, 2, remat={0: 1})
    print("\nK=2 without marker  -> slots:", sorted(plain.slots))
    print("K=2 with remat={0:1} -> slots:", sorted(marked.slots),
          " rematerialized:", marked.rematerialized)
    assert 1 in plain.slots and 1 in marked.rematerialized
    assert 1 not in marked.slots
    print("\nlowered code (notice uid0 is elided; 'remat' appears at uid8):")
    print(marked.allocated_program.render())
    res = equivalence_check(prog, marked)
    print("\nsymbolic before/after equivalence:", res.summary())
    assert res.equivalent, res.failure


def show_rejection_boundary() -> None:
    hr("3) ACTUALLY TRIGGERED REJECTION / EXCEPTION BOUNDARIES")

    print("(a) proven infeasible — two operands pinned to the same register:")
    prog = sample_fixed_collision()
    try:
        allocate(prog, 3)
    except InfeasibleProgramError as exc:
        print(f"    InfeasibleProgramError: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected InfeasibleProgramError")

    print("\n(b) strict input — bool is not accepted where K:int is required:")
    prog = sample_register_pressure()
    try:
        allocate(prog, True)  # type: ignore[arg-type]
    except ProgramFormatError as exc:
        print(f"    ProgramFormatError: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected ProgramFormatError")

    print("\n(c) strict input — K outside the contracted range 2..8:")
    try:
        allocate(prog, 9)
    except ProgramFormatError as exc:
        print(f"    ProgramFormatError: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected ProgramFormatError")

    print("\nBudget exhaustion (BudgetExhaustedError) is a distinct class from")
    print("the proven-infeasibility shown above; see tests/test_allocation.py.")


def main() -> None:
    t0 = time.perf_counter()
    show_normal_spill()
    show_call_clobber()
    show_rematerialisation()
    show_rejection_boundary()
    hr("DEMO COMPLETE")
    print(f"total wall time: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
