"""Fixed-input demonstration for the register allocator.

Run with::

    python demo.py

It performs real allocation (no sleeps, no pre-recorded output) and prints:

1. a normal result on a program that genuinely requires spilling (values
   live across a call that clobbers registers, K=2), including the emitted
   machine code, spill slots and an independent equivalence check;
2. an actual rejection boundary: the same call on a program that is
   *provably* impossible (a value forced into a register the call destroys)
   is rejected with UnsatisfiableProgram, which is distinct from the
   search-budget exception.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regalloc import (  # noqa: E402
    AllocationBudgetExhausted,
    Instruction,
    Program,
    Block,
    UnsatisfiableProgram,
    allocate,
)
from regalloc.examples import call_pressure_program  # noqa: E402
from tests.verifier import verify_allocation  # noqa: E402


def _loc(loc) -> str:
    return f"r{loc.i}" if hasattr(loc, "i") and type(loc).__name__ == "Reg" else f"[sp+{loc.i}]"


def print_program(title: str, program: Program) -> None:
    print(title)
    if program.params:
        print(f"  params: {', '.join(program.params)}")
    for b in program.blocks:
        print(f"  block {b.id} -> {', '.join(b.succs) or '(exit)'}")
        for phi in b.phis:
            src = ", ".join(f"{p}:{s}" for p, s in sorted(phi.srcs.items()))
            fix = f" [fixed r{phi.fixed_def}]" if phi.fixed_def is not None else ""
            print(f"      phi {phi.dst} = ({src}){fix}")
        for i, ins in enumerate(b.instrs):
            fix = ""
            if ins.fixed_use or ins.fixed_def:
                fu = ", ".join(f"{v}->r{r}" for v, r in sorted(ins.fixed_use.items()))
                fd = ", ".join(f"{v}->r{r}" for v, r in sorted(ins.fixed_def.items()))
                fix = f" [fixed use:{fu} def:{fd}]"
            clob = f" clobber {sorted(ins.clobbers)}" if ins.clobbers else ""
            print(f"    {i}: {ins.opcode} {list(ins.uses)} -> {list(ins.defs)}{fix}{clob}")


def print_result(result) -> None:
    print("  register assignment (spilled variables -> None):")
    for v, r in result.assignment.items():
        print(f"      {v:>4} : {'spilled' if r is None else 'r' + str(r)}")
    if result.stack_slots:
        print("  stack slots (assigned by sorted variable id):")
        for v in sorted(result.stack_slots):
            print(f"      {v:>4} : [sp+{result.stack_slots[v]}]")
    print(f"  spill/recolor iterations: {result.iterations}")
    print("  emitted machine code:")
    for mb in result.program.blocks:
        print(f"    block {mb.id} -> {', '.join(mb.succs) or '(exit)'}")
        for mi in mb.instrs:
            r = " ".join(_loc(x) for x in mi.reads)
            w = " ".join(_loc(x) for x in mi.writes)
            extra = f"  clobber {sorted(mi.clobbers)}" if mi.clobbers else ""
            print(f"        {mi.opcode:6s} reads=({r}) writes=({w}){extra}")


def main() -> int:
    t0 = time.time()

    print("=" * 72)
    print("PART 1 (normal result): live-across-call pressure forces a spill")
    print("=" * 72)
    program = call_pressure_program()
    print_program("Input program (K = 2; the call clobbers r0 and r1):", program)

    result = allocate(program, K=2)
    print_result(result)

    spilled = [v for v, r in result.assignment.items() if r is None]
    assert spilled, "demo program was expected to require at least one spill"
    verify_allocation(program, result)
    print("  independent symbolic verifier: EQUIVALENT (operands/results/liveness)")

    print()
    print("=" * 72)
    print("PART 2 (rejection boundary): provably impossible fixed register")
    print("=" * 72)
    # `a` is forced into r1 and is also used after the call, which clobbers
    # r1. Spilling cannot repair this: the *use at the call itself* must be
    # in r1 while the call destroys r1, yet the value is needed afterwards.
    impossible = Program(
        entry="entry",
        params=("a",),
        blocks=(
            Block(
                "entry",
                instrs=(
                    Instruction(
                        "call",
                        ("a",),
                        ("c",),
                        fixed_use={"a": 1},
                        clobbers=(1,),
                    ),
                    Instruction("ret", ("a", "c"), ()),
                ),
                succs=(),
            ),
        ),
    )
    print_program("Input program (K = 2):", impossible)
    print("  calling allocate(...) ->")
    try:
        allocate(impossible, K=2)
    except UnsatisfiableProgram as exc:
        print(f"      UnsatisfiableProgram: {exc}")
        print("      (this is a proof of impossibility, not search-budget "
              "exhaustion)")
    except AllocationBudgetExhausted:  # pragma: no cover - would be a bug
        print("      ERROR: budget exhausted reported for a provably unsat program")
        return 1
    else:  # pragma: no cover - would be a bug
        print("      ERROR: an impossible program was accepted")
        return 1

    print(f"\ndemo finished in {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
