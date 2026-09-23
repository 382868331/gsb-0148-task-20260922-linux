"""Offline demonstration of the v2 backend: register views, exceptional
control flow, early-clobber, no-free-temp copy cycles and stack slots.

Run with::

    python demo2.py

Everything shown is computed now (nothing is precomputed or fetched).  It
prints:

1. a normal allocation of the combined torture CFG — exceptional edge,
   early-clobber op, double-width value across a call and a copy cycle on
   an edge with no free temporary — with independent verification and the
   paired bit-level symbolic-execution result (normal and exceptional exits);
2. the tricky boundary: which mechanism broke the copy cycle (register temp
   vs scratch stack slot), plus stack-slot reuse with alignment;
3. two actually triggered refusal paths: a proven infeasibility (pinned
   operands in aliasing views) and a budget-exhausted explicit unallocated
   status;
4. the non-toy main-scale program (>=100 blocks / >=1000 instrs / 16 views).
"""

from __future__ import annotations

import time

from regalloc import allocate2, verify_allocation
from regalloc.model import Block, Instr, Program
from regalloc.samples2 import (
    generate_scale_program,
    machine_16,
    machine_reuse,
    machine_torture,
    sample_slot_reuse,
    sample_v2_torture,
)
from regalloc.symbolic2 import equivalence_check2


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_torture() -> None:
    hr("1) NORMAL RESULT — combined torture CFG (views, exception, call)")
    prog = sample_v2_torture()
    machine = machine_torture()
    print("program:", prog.name, "| blocks:", len(prog.blocks),
          "| views:", len(machine.views),
          "| caller-saved:", [v.name for v in machine.views
                              if not v.callee_saved])
    n_throws = sum(1 for b in prog.blocks for i in b.instrs if i.may_throw)
    print("may_throw instructions:", n_throws,
          "| exceptional edges:",
          [(i, q) for i, b in enumerate(prog.blocks) for q in b.exc_succs])
    print("early-clobber defs:",
          [(i.uid, i.early_clobber) for b in prog.blocks for i in b.instrs
           if i.early_clobber])

    t0 = time.perf_counter()
    result = allocate2(prog, machine)
    elapsed = time.perf_counter() - t0
    print(f"\nallocated in {elapsed*1000:.1f} ms; status={result.status}; "
          f"spill iterations={result.iterations}")
    assert result.allocated, result.message

    print("home views:", {f"v{v}": view for v, view in result.assignment.items()})
    print("stack slots:", {f"v{v}": f"[{s.offset},{s.offset + s.size})"
                           for v, s in result.slots.items()},
          "frame:", result.frame_size, "bytes")

    # the wide x-class value survives the may_throw call in a callee-saved view
    wide = result.assignment[3]
    print(f"double-width v3 survives the call in {wide} "
          f"(callee_saved={machine.view(wide).callee_saved})")

    report = verify_allocation(prog, machine, result)
    print("independent per-edge verifier:", report.summary())
    assert report.ok, report.failures

    eq = equivalence_check2(prog, machine, result, depth_bound=30,
                            step_budget=20_000)
    print("paired bit-level symbolic executor:", eq.summary())
    assert eq.equivalent, eq.failure

    print("\nlowered machine code (exceptional split block included):")
    print(result.allocated_program.render())


def show_tricky_boundaries() -> None:
    hr("2) TRICKY BOUNDARIES — cycle breaking and stack-slot reuse")
    prog = sample_v2_torture()
    result = allocate2(prog, machine_torture())
    print("copy-cycle mechanisms per edge:",
          {f"{p}->{q}": breaks
           for (p, q), breaks in result.allocated_program.edge_breaks.items()})
    print("  edge 1->1 is a loop back edge: it swaps the only two s-class")
    print("  views while x/w values occupy every other register group, so NO")
    print("  free temporary view exists; the cycle is broken through the")
    print("  dedicated scratch STACK slot (never a live alias location).")

    print()
    reuse_prog = sample_slot_reuse()
    r2 = allocate2(reuse_prog, machine_reuse())
    by_offset: dict[int, list[int]] = {}
    for v, slot in r2.slots.items():
        by_offset.setdefault(slot.offset, []).append(v)
    print("slot-reuse example — spilled variables with disjoint live ranges")
    print("share stack bytes (offset -> variables):")
    for off in sorted(by_offset):
        vs = by_offset[off]
        size = r2.slots[vs[0]].size
        shared = "  <-- REUSED (disjoint live ranges)" if len(vs) > 1 else ""
        print(f"  [{off},{off + size}): variables {sorted(vs)}{shared}")
    assert any(len(vs) > 1 for vs in by_offset.values())
    rep = verify_allocation(reuse_prog, machine_reuse(), r2)
    assert rep.ok, rep.failures
    print("independent verifier:", rep.summary())


def show_refusals() -> None:
    hr("3) ACTUALLY TRIGGERED REFUSALS (no seemingly-complete output)")

    print("(a) proven infeasible — two operands pinned into ALIASING views")
    print("    (w0 covers bits 0..31 of group 0; x0 covers bits 0..63):")
    prog = Program(
        blocks=(Block(label="e", instrs=(
            Instr(uid=0, uses=(0, 1), defs=(),
                  fixed_views={0: "w0", 1: "x0"}),
        ),),),
        params=(0, 1),
        var_classes={0: "w", 1: "x"},
    )
    r = allocate2(prog, machine_torture())
    print("    status:", r.status)
    print("    message:", r.message)
    print("    allocated_program is None:", r.allocated_program is None)
    assert r.status == "infeasible"
    assert r.allocated_program is None

    print("\n(b) budget exhausted — feasibility UNKNOWN, explicit state:")
    r2 = allocate2(sample_v2_torture(), machine_torture(),
                   coloring_budget=1)
    print("    status:", r2.status)
    print("    message:", r2.message)
    print("    allocated_program is None:", r2.allocated_program is None,
          "| assignment is None:", r2.assignment is None)
    rep = verify_allocation(sample_v2_torture(), machine_torture(), r2)
    print("    verifier on this result:", rep.summary())
    assert r2.status == "budget_exhausted"
    assert r2.allocated_program is None
    assert rep.ok


def show_scale() -> None:
    hr("4) MAIN SCALE — deterministic non-toy program, 16 register views")
    prog = generate_scale_program()
    machine = machine_16()
    n_instr = sum(len(b.instrs) for b in prog.blocks)
    print(f"blocks={len(prog.blocks)} instructions={n_instr} views=16 "
          f"(8x 32-bit + 8x 64-bit aliasing pairs)")
    t0 = time.perf_counter()
    result = allocate2(prog, machine)
    elapsed = time.perf_counter() - t0
    print(f"status={result.status} iterations={result.iterations} "
          f"spilled={len(result.slots or {})} frame={result.frame_size} "
          f"in {elapsed:.1f} s")
    assert result.allocated, result.message
    rep = verify_allocation(prog, machine, result)
    print("independent verifier:", rep.summary())
    assert rep.ok
    eq = equivalence_check2(prog, machine, result,
                            depth_bound=90, step_budget=25_000)
    print("bit-level symbolic executor:", eq.summary())
    print("(bounded exploration; full extended run uses higher bounds, "
          "see README)")
    assert eq.equivalent, eq.failure


def main() -> None:
    t0 = time.perf_counter()
    show_torture()
    show_tricky_boundaries()
    show_refusals()
    show_scale()
    hr("DEMO COMPLETE")
    print(f"total wall time: {time.perf_counter() - t0:.1f} s "
          f"(budget: 90 s)")


if __name__ == "__main__":
    main()
