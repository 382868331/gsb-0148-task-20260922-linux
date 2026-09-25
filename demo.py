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
from regalloc.concrete import exit_live_vars, run_allocated, run_source
from regalloc.liveness import liveness_with_phis
from regalloc.profile import edge_frequencies, memory_traffic
from regalloc.samples import (
    COLD_PARAMS,
    COLD_PROFILE,
    HOT_PARAMS,
    HOT_PROFILE,
    sample_call_surviving,
    sample_fixed_collision,
    sample_hot_loop_cold_error,
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

    print("\n(d) strict input — frequency profile contract (block count / "
          "bool / negative):")
    hotprog = sample_hot_loop_cold_error()
    for bad in ((1, 2, 3), (0, 0, 0, False, 0), (1, -1, 1, 1, 1)):
        try:
            allocate(hotprog, 4, frequencies=bad)
        except ProgramFormatError as exc:
            print(f"    frequencies={bad!r}: {exc}")
        else:  # pragma: no cover
            raise AssertionError("expected ProgramFormatError")

    print("\nBudget exhaustion (BudgetExhaustedError) is a distinct class from")
    print("the proven-infeasibility shown above; see tests/test_allocation.py.")


def show_frequency_guided_spills() -> None:
    hr("4) FREQUENCY-GUIDED SPILLS — hot loop with a cold error arm, K=4")
    prog = sample_hot_loop_cold_error()
    print("program:", prog.name, "| blocks: entry, loop, latch, cold, exit")
    print("profiles (per block):")
    print("  HOT  :", HOT_PROFILE, "  -> error arm is rare (~1/1000)")
    print("  COLD :", COLD_PROFILE, "  -> exact inversion: error arm is hot")

    legacy = allocate(prog, 4)
    hot = allocate(prog, 4, frequencies=HOT_PROFILE)
    cold = allocate(prog, 4, frequencies=COLD_PROFILE)

    print("\nspilled variables (slot == variable id):")
    print(f"  legacy (no profile): {sorted(legacy.slots)}")
    print(f"  HOT  profile       : {sorted(hot.slots)}")
    print(f"  COLD profile       : {sorted(cold.slots)}")
    print("  per-iteration victim sequences (spill iteration rebuilt liveness "
          "and costs):")
    for name, a in (("legacy", legacy), ("HOT", hot), ("COLD", cold)):
        victims = []
        prev: set = set()
        for state in a.history:
            victims.extend(sorted(set(state) - prev))
            prev = set(state)
        print(f"    {name:6s} -> {victims}")

    print("\nACTUAL weighted reload/store traffic (counted in the lowered")
    print("machine instructions, NOT from the heuristic cost tables):")
    rows = []
    for pname, profile in (("HOT", HOT_PROFILE), ("COLD", COLD_PROFILE)):
        ef = edge_frequencies(prog, profile)
        for aname, a in (("legacy", legacy), ("HOT", hot), ("COLD", cold)):
            t = memory_traffic(prog, a.allocated_program, profile, ef)
            rows.append((pname, aname, t))
            print(f"  weights={pname:<4s} allocation={aname:<6s} "
                  f"raw loads={t['loads']:2d} stores={t['stores']:2d}   "
                  f"weighted loads={t['weighted_loads']:5d} "
                  f"stores={t['weighted_stores']:5d} "
                  f"TOTAL={t['weighted_total']:5d}")

    def total_for(a, profile):
        ef = edge_frequencies(prog, profile)
        return memory_traffic(prog, a.allocated_program, profile, ef)[
            "weighted_total"]

    wh_leg, wh_hot = total_for(legacy, HOT_PROFILE), total_for(hot, HOT_PROFILE)
    wc_leg, wc_cold = total_for(legacy, COLD_PROFILE), total_for(cold, COLD_PROFILE)
    print(f"\n  HOT profile:  legacy {wh_leg} -> profiled {wh_hot}  "
          f"({100.0 * (wh_leg - wh_hot) / wh_leg:.1f}% less hot memory traffic)")
    print(f"  COLD profile: legacy {wc_leg} -> profiled {wc_cold} "
          f"({100.0 * (wc_leg - wc_cold) / wc_leg:.1f}% less traffic on the "
          f"now-hot error arm)")

    print("\nper-site weighted traffic of the HOT allocation under HOT profile:")
    ef = edge_frequencies(prog, HOT_PROFILE)
    for site in memory_traffic(prog, hot.allocated_program, HOT_PROFILE, ef)[
            "by_site"]:
        print(f"    {site['site']:<14s} freq={site['frequency']:5d} "
              f"loads={site['loads']} stores={site['stores']} "
              f"weighted={site['weighted_loads'] + site['weighted_stores']}")

    # ---- independent concrete equivalence on two real executions ---------
    print("\nconcrete before/after equivalence (independent tiny interpreters):")
    for pname, params in (("hot-path input", HOT_PARAMS),
                          ("error-path input", COLD_PARAMS)):
        src = run_source(prog, params)
        observable = exit_live_vars(prog, src.exit_block)
        print(f"  {pname} {params}: {len(src.block_trace)} block visits, "
              f"cold arm visited "
              f"{sum(1 for b in src.block_trace if b == 3)} time(s)")
        for aname, a in (("legacy", legacy), ("HOT", hot), ("COLD", cold)):
            got = run_allocated(prog, a, params)
            assert got.uid_trace == src.uid_trace
            assert got.block_trace == src.block_trace
            assert all(got.values[v] == src.values[v] for v in observable)
            vals = ", ".join(f"v{v}={got.values[v]}" for v in sorted(observable))
            print(f"    {aname:6s}: identical uid/block traces and results ({vals})")

    for aname, a in (("legacy", legacy), ("HOT", hot), ("COLD", cold)):
        res = equivalence_check(prog, a, depth_bound=12, step_budget=20_000)
        assert res.equivalent, res.failure
        print(f"  symbolic bounded cross-check {aname:6s}: {res.summary()}")

    print("\nlowered code of the HOT allocation — the spilled cold seed v2")
    print("and the cold-arm temporaries only ever touch memory inside the")
    print("cold block (slot2/slot9..12); the hot loop/latch blocks carry no")
    print("loads or stores of them:")
    print(hot.allocated_program.render())


def main() -> None:
    t0 = time.perf_counter()
    show_normal_spill()
    show_call_clobber()
    show_frequency_guided_spills()
    show_rejection_boundary()
    hr("DEMO COMPLETE")
    print(f"total wall time: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
