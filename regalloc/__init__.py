"""Register allocation with call-clobber constraints (public package).

Public surface::

    allocate(program, K) -> Allocation
    Program / Block / Instr dataclasses
    program_from_dict(...)
    exception hierarchy: RegAllocError, ProgramFormatError,
                          InfeasibleProgramError, BudgetExhaustedError
"""

from .model import (
    Block,
    Instr,
    Program,
    program_from_dict,
)
from .allocator import (
    Allocation,
    RegAllocError,
    ProgramFormatError,
    InfeasibleProgramError,
    BudgetExhaustedError,
    RematInfo,
    allocate,
)
from .liveness import LivenessReport

__all__ = [
    "allocate",
    "Allocation",
    "LivenessReport",
    "Program",
    "Block",
    "Instr",
    "program_from_dict",
    "RegAllocError",
    "ProgramFormatError",
    "InfeasibleProgramError",
    "BudgetExhaustedError",
    "RematInfo",
]
