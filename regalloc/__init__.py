"""regalloc -- register allocation with call-clobber constraints.

Public API
----------
:func:`allocate` is the only entry point::

    from regalloc import (
        allocate, Program, Block, Phi, Instruction,
        Allocation, AllocatedProgram, Liveness,
        Reg, Slot,
        AllocatorError, InvalidArgumentError, InvalidProgramError,
        UnsatisfiableProgram, AllocationBudgetExhausted,
    )
    result: Allocation = allocate(program, K=4)

The returned :class:`~regalloc.model.Allocation` carries the concrete
:class:`~regalloc.model.AllocatedProgram`, the virtual-variable -> register
assignment (``None`` for spilled variables), stack slot indices, the
independently checkable edge-sensitive :class:`~regalloc.model.Liveness`
sets of the ORIGINAL program, and the number of spill iterations used.

Only the Python 3 standard library is required (target: CPython 3.14.7).
"""

from .allocator import allocate
from .model import (
    AllocatedProgram,
    Allocation,
    AllocationBudgetExhausted,
    AllocatorError,
    Block,
    Instruction,
    InvalidArgumentError,
    InvalidProgramError,
    Liveness,
    MBlock,
    MInstr,
    Phi,
    Program,
    Reg,
    Slot,
    UnsatisfiableProgram,
)
from .validate import check_finite_number

__all__ = [
    "allocate",
    "Program",
    "Block",
    "Phi",
    "Instruction",
    "Allocation",
    "AllocatedProgram",
    "MBlock",
    "MInstr",
    "Liveness",
    "Reg",
    "Slot",
    "AllocatorError",
    "InvalidArgumentError",
    "InvalidProgramError",
    "UnsatisfiableProgram",
    "AllocationBudgetExhausted",
    "check_finite_number",
]
