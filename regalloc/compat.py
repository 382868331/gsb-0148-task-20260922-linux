"""Compatibility adapter: run legacy v1 programs through the v2 pipeline.

The v1 entry point :func:`regalloc.allocate` is unchanged and remains the
primary way to allocate legacy programs.  This module exists so callers
who *want* the v2 result shape (status object, named views, stack-slot
layout) for a legacy program have a documented migration path:

* a legacy ``Program`` (no ``var_classes``) is converted to a v2 program
  by giving every variable the single class ``"gpr"`` and rewriting
  integer ``fixed``/``clobbers`` to named views of
  :func:`regalloc.machine.flat_machine`;
* the converted program is allocated with :func:`regalloc.allocate2`.

The mapping is semantics-preserving: the flat machine has K independent
32-bit views and no callee-saved registers, so the only clobbers are the
explicit per-instruction ones, exactly as in v1.
"""

from __future__ import annotations

from .allocator2 import AllocationResult, allocate2
from .machine import flat_machine
from .model import Block, Instr, Program, check_int, MAX_K, MIN_K, ProgramFormatError


def to_v2_program(program: Program) -> Program:
    """Convert a legacy program to an equivalent v2 program."""
    if not isinstance(program, Program):
        raise ProgramFormatError("to_v2_program: expected a Program instance")
    if program.var_classes is not None:
        raise ProgramFormatError("to_v2_program: program is already v2")
    blocks = []
    for b in program.blocks:
        instrs = []
        for ins in b.instrs:
            instrs.append(
                Instr(
                    uid=ins.uid,
                    uses=ins.uses,
                    defs=ins.defs,
                    fixed_views={v: f"r{r}" for v, r in ins.fixed.items()},
                    clobber_views=tuple(f"r{r}" for r in ins.clobbers),
                    note=ins.note,
                )
            )
        blocks.append(
            Block(label=b.label, instrs=tuple(instrs), succs=b.succs, phis=b.phis)
        )
    all_vars = set(program.params)
    for b in program.blocks:
        for ins in b.instrs:
            all_vars.update(ins.defs)
        for phi in b.phis:
            all_vars.add(phi.def_var)
    return Program(
        blocks=tuple(blocks),
        params=program.params,
        name=program.name,
        var_classes={v: "gpr" for v in all_vars},
    )


def allocate_legacy_via_v2(program: Program, k: int) -> AllocationResult:
    """Allocate a legacy program through the v2 pipeline (migration aid)."""
    check_int(k, "K", minimum=MIN_K)
    if k > MAX_K:
        raise ProgramFormatError(f"K: expected integer <= {MAX_K}, got {k}")
    machine = flat_machine(k)
    return allocate2(to_v2_program(program), machine)
