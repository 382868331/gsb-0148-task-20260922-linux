"""Offline demonstration: register allocation, versions 1 and 2.

Run with::

    python demo.py

It performs real allocations on deterministic in-repo programs (nothing is
precomputed, fetched or sleep-padded).  It prints:

1. (v1) a normal K=2 allocation that spills, with the old numeric-register
   contract — kept working unchanged;
2. (v2) a normal allocation using the NEW capabilities: width-aliased views,
   a double-width value surviving a call in callee-saved banks, an
   early-clobber op and an exception edge, with the independent bit-level
   per-edge checker result;
3. a TRICKY BOUNDARY: a true register swap cycle on the 2-bank target where
   NO free temporary register exists — the cycle is broken with a private
   stack slot (never a live aliased view);
4. an actually triggered REJECTION (proven early-clobber infeasibility) and
   a budget-exhaustion case (feasibility unknown, no machine program);
5. a MAIN-SCALE run: a generated 110-block / ~1100-instruction / 16-view
   CFG, allocated and checked on EVERY edge (normal and exception exits).

Total wall time is printed and is well under the 90 s budget.
"""

from __future__ import annotations

import time

from regalloc import (
    BudgetExhaustedError,
    InfeasibleProgramError,
    ProgramFormatError,
    allocate,
)
from regalloc.samples import sample_register_pressure
from regalloc.liveness import liveness_with_phis
from regalloc.symbolic import equivalence_check
from regalloc.samples2 import (
    sample_combined,
    sample_copy_cycle_no_temp,
    sample_mega,
    sample_pair_across_call,
    generate_large_v2,
)
from regalloc import Block, Instr, Program, default_target


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_v1_spill() -> None:
    hr("1) FORMAT v1 — normal K=2 allocation that spills (unchanged contract)")
    prog = sample_register_pressure()
    t0 = time.perf_counter()
    alloc = allocate(prog, 2)
    print(f"allocated in {(time.perf_counter()-t0)*1000:.1f} ms; "
          f"iterations={alloc.iterations}, slots={sorted(alloc.slots)}")
    res = equivalence_check(prog, alloc)
    print("v1 symbolic before/after equivalence:", res.summary())
    assert res.equivalent, res.failure


def show_v2_new_capabilities() -> None:
    hr("2) FORMAT v2 — ONE CFG: pair-across-call + early-clobber + exception + "
       "no-free-temp swap")
    prog = sample_mega()
    t0 = time.perf_counter()
    a = allocate(prog)
    dt = time.perf_counter() - t0
    print(f"program: {prog.name} | blocks={len(prog.blocks)} | "
          f"views={len(prog.target.views)} | {dt*1000:.1f} ms")
    print("double-width value v1 stays register-resident at:", a.assignment[1],
          "(ebx,edx = the callee-saved pair across the eax/ecx-clobbering call)")
    print("join phi homes: u ->", a.assignment.get(7), " v ->", a.assignment.get(8))
    print("scratch stack slot used on the join edge:", a.scratch_slots)
    print("(all four banks were full, so the eax<->ecx swap had no free",
          "temporary register)")
    vr = a.verification
    print(f"independent bit-level checker: ok={vr.ok}")
    print(f"  edges checked (EVERY CFG edge): {vr.edges_checked}")
    print(f"  normal exits={vr.normal_exits}  exception exits={vr.exception_exits}")
    assert vr.ok, vr.failure_text()

    print("\njoin edge machine code (cycle broken via the private stack slot):")
    for b in a.allocated_program.blocks:
        if b.split_edge == (0, 1):
            for mi in b.instrs:
                print("   ", mi.render())

    print("\nmachine code of the pre block (call, early-clobber, risky op):")
    for b in a.allocated_program.blocks:
        if b.origin == 0 and b.exc_instr is None:
            for mi in b.instrs:
                print("   ", mi.render())


def show_no_temp_cycle() -> None:
    hr("3) TRICKY BOUNDARY (small target) — swap cycle on a 2-bank machine")
    prog = sample_copy_cycle_no_temp()
    a = allocate(prog)
    print("target: 2 banks (t0 caller-saved, t1 callee-saved); both are full.")
    print("scratch slots reported:", a.scratch_slots)
    for b in a.allocated_program.blocks:
        if b.split_edge == (0, 1):
            print("edge 0->1 machine code:")
            for mi in b.instrs:
                print("   ", mi.render())
    assert a.scratch_slots, "this case must use a stack scratch slot"
    assert a.verification.ok, a.verification.failure_text()
    print("per-edge bit-level checker: ok, every edge verified.")


def show_rejections() -> None:
    hr("4) ACTUALLY TRIGGERED REJECTION / FAILURE BOUNDARIES")

    print("(a) proven infeasible — early-clobber result pinned to the same")
    print("    view as a surviving fixed input at the same instruction:")
    t = default_target()
    bad = Program(blocks=(Block(label="b", instrs=(
        Instr(uid=0, uses=(), defs=(1,)),
        Instr(uid=1, uses=(), defs=(2,)),
        Instr(uid=2, uses=(1,), defs=(3, 4), early_clobber=(3,),
              fixedv={1: "eax", 3: "eax"}),
        Instr(uid=3, uses=(1, 4), defs=(5,)),
        Instr(uid=4, uses=(5,), defs=()),
    )),), version=2, target=t)
    try:
        allocate(bad)
    except InfeasibleProgramError as exc:
        print("    InfeasibleProgramError:", str(exc)[:120])
    else:  # pragma: no cover
        raise AssertionError("expected InfeasibleProgramError")

    print("\n(b) budget exhausted is 'unknown', never a fake complete program:")
    import regalloc.allocator2 as mod
    saved = mod.COLORING_BUDGET_V2
    mod.COLORING_BUDGET_V2 = 0
    try:
        allocate(sample_combined())
    except BudgetExhaustedError as exc:
        print("    BudgetExhaustedError:", str(exc)[:120])
    else:  # pragma: no cover
        raise AssertionError("expected BudgetExhaustedError")
    finally:
        mod.COLORING_BUDGET_V2 = saved

    print("\n(c) strict input — passing numeric K to a version=2 program:")
    try:
        allocate(sample_pair_across_call(), 4)
    except ProgramFormatError as exc:
        print("    ProgramFormatError:", exc)
    else:  # pragma: no cover
        raise AssertionError("expected ProgramFormatError")


def show_main_scale() -> None:
    hr("5) MAIN SCALE — 110 blocks / ~1100 instrs / 16 views (seeded, real run)")
    prog = generate_large_v2()
    n_instrs = sum(len(b.instrs) for b in prog.blocks)
    print(f"generated {len(prog.blocks)} blocks, {n_instrs} instructions, "
          f"{len(prog.target.views)} views")
    t0 = time.perf_counter()
    a = allocate(prog)
    dt = time.perf_counter() - t0
    vr = a.verification
    total_edges = sum(len(b.succs) for b in prog.blocks)
    print(f"allocated + independently verified in {dt:.2f} s")
    print(f"  spill iterations={a.iterations}, spilled={len(a.slots)}, "
          f"frame units={a.frame_slots}")
    print(f"  checker: ok={vr.ok}, edges={vr.edges_checked}/{total_edges}, "
          f"normal exits={vr.normal_exits}, exception exits={vr.exception_exits}")
    assert vr.ok, vr.failure_text()
    assert vr.edges_checked == total_edges


def main() -> None:
    t0 = time.perf_counter()
    show_v1_spill()
    show_v2_new_capabilities()
    show_no_temp_cycle()
    show_rejections()
    show_main_scale()
    hr("DEMO COMPLETE")
    print(f"total wall time: {time.perf_counter() - t0:.2f} s (budget 90 s)")


if __name__ == "__main__":
    main()
