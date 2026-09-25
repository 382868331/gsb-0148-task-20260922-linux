"""Offline demonstration: register allocation with call-clobber constraints.

Run with::

    ./.venv/bin/python demo.py

It performs real allocations on deterministic in-repo programs (nothing is
precomputed or fetched).  It prints:

1. a normal allocation that genuinely spills at K=2, with the register/stack
   map, independently verifiable edge-live sets, the lowered machine code and
   the paired symbolic-execution equivalence result;
2. a call-clobber case (a value surviving a call);
3. the block-frequency-aware spill policy on a hot loop with a cold
   error-recovery branch: run once with the normal (hot loop / cold branch)
   profile and once with the inverted profile, with the paired executor
   confirming before/after equivalence each time, a hand layout recount of
   the *actually inserted* weighted reload/store traffic, and a comparison
   against the original frequency-free policy;
4. an actually triggered rejection boundary (provably infeasible fixed-
   register collision), plus a strict-input rejection (bool as K).
"""

from __future__ import annotations

import time

from regalloc import (
    InfeasibleProgramError,
    Program,
    ProgramFormatError,
    actual_access_stats,
    allocate,
    spill_cost_table,
)
from regalloc.liveness import liveness_with_phis
from regalloc.samples import (
    sample_call_surviving,
    sample_fixed_collision,
    sample_hot_cold_loop,
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


def _hand_recount(prog, alloc):
    """Independent ground-truth recount, written out here rather than reusing
    alloc.access_stats: walk the lowered blocks, weight each load/store by the
    lowered block's frequency (origin block, or predecessor for a split)."""
    base = prog.frequencies
    wl = ws = nl = ns = 0
    for b in alloc.allocated_program.blocks:
        if base is None:
            f = 1
        elif b.origin >= 0:
            f = base[b.origin]
        else:
            f = base[b.split_edge[0]]
        for mi in b.instrs:
            if mi.kind == "load":
                nl += 1
                wl += f
            elif mi.kind == "store":
                ns += 1
                ws += f
    return nl, ns, wl, ws


def show_frequency_policy() -> None:
    hr("3) FREQUENCY-AWARE SPILL POLICY — hot loop with a cold error branch")
    hot = sample_hot_cold_loop()
    freqs = hot.frequencies
    labels = [b.label for b in hot.blocks]
    print("program:", hot.name, "(K=3)")
    print("blocks:", labels)
    print("per-block execution-frequency hints:", dict(zip(labels, freqs)))
    print("(100 = hot loop header/body/latch; 0 = cold error recovery which is")
    print(" structurally INSIDE the natural loop, so loop-depth weighting cannot")
    print(" tell it apart from the hot blocks)")

    # ---- heuristic cost table for the live-across-loop candidates --------
    table = spill_cost_table(hot, freqs)
    print("\npredicted weighted reload/store cost if each candidate is spilled")
    print("(heuristic estimate from real use/def + edge phi points):")
    for v in (1, 2, 3):
        c = table[v]
        print(f"  v{v}: static loads/stores={c.loads}/{c.stores} "
              f"weighted={c.weighted_total} "
              f"(wl={c.weighted_loads}, ws={c.weighted_stores})")

    # ---- original policy: same shape, no frequencies ---------------------
    old_shape = Program(blocks=hot.blocks, params=hot.params, frequencies=None)
    old_alloc = allocate(old_shape, 3)
    print("\n(a) ORIGINAL policy (no frequencies), spill decisions:")
    for d in old_alloc.spill_decisions:
        print(f"      round {d.iteration}: spill v{d.victim} "
              f"(degree+1)/weighted-use = {d.heuristic_score:.3f}")
    print(f"    spilled: {sorted(old_alloc.slots)}")
    old_under_hot = actual_access_stats(hot, old_alloc)
    print(f"    ACTUAL traffic emitted: {old_under_hot.row()}")
    print("    (static loads/stores stay fixed; weighted column re-evaluates")
    print("     that exact lowering against the hot profile)")

    # ---- frequency policy, normal profile --------------------------------
    new_alloc = allocate(hot, 3)
    print("\n(b) FREQUENCY policy with the hot profile, spill decisions:")
    for d in new_alloc.spill_decisions:
        print(f"      round {d.iteration}: spill v{d.victim} "
              f"predicted weighted cost = {d.weighted_cost}")
    print(f"    spilled: {sorted(new_alloc.slots)}")
    stats = new_alloc.access_stats
    print(f"    ACTUAL traffic emitted: {stats.row()}")

    nl, ns, wl, ws = _hand_recount(hot, new_alloc)
    print(f"    hand recount of the lowered machine code: "
          f"loads={nl} stores={ns} weighted(loads={wl}, stores={ws}, "
          f"total={wl + ws})")
    assert (nl, ns, wl, ws) == (
        stats.loads, stats.stores, stats.weighted_loads, stats.weighted_stores
    ), "hand recount disagrees with alloc.access_stats"
    print("    -> hand recount matches access_stats exactly.")

    res = equivalence_check(hot, new_alloc, depth_bound=24, step_budget=30000)
    print("    before/after symbolic equivalence:", res.summary())
    assert res.equivalent, res.failure

    saved = old_under_hot.weighted_total - stats.weighted_total
    print(f"\n    weighted memory accesses on the common path:")
    print(f"      original policy lowering: {old_under_hot.weighted_total}")
    print(f"      frequency policy lowering: {stats.weighted_total}")
    print(f"      reduction: {saved} "
          f"({100.0 * saved / old_under_hot.weighted_total:.0f}% fewer "
          f"weighted spill accesses)")
    print("    the frequency policy keeps the doubly-used hot invariant v3 in")
    print("    a register, spills v1 (only one reload per loop visit) and moves")
    print("    the bulk of the spill traffic onto the never-taken cold block.")

    # ---- inverted profile: same shape, cold/hot swapped ------------------
    inv = sample_hot_cold_loop(inverted=True)
    print("\n(c) INVERTED profile (recovery block is now hot, loop body cold):")
    print("    per-block hints:", dict(zip(labels, inv.frequencies)))
    inv_alloc = allocate(inv, 3)
    print("    spill decisions:",
          [(f"v{d.victim}", d.weighted_cost) for d in inv_alloc.spill_decisions])
    print(f"    spilled: {sorted(inv_alloc.slots)}  {inv_alloc.access_stats.row()}")
    res = equivalence_check(inv, inv_alloc, depth_bound=24, step_budget=30000)
    print("    before/after symbolic equivalence:", res.summary())
    assert res.equivalent, res.failure
    print("    -> the decision reverses: v2 is now kept in a register and the")
    print("       cold old-loop value v3 is spilled, as the profile demands.")


def show_rejection_boundary() -> None:
    hr("4) ACTUALLY TRIGGERED REJECTION / EXCEPTION BOUNDARIES")

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
    show_frequency_policy()
    show_rejection_boundary()
    hr("DEMO COMPLETE")
    print(f"total wall time: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
