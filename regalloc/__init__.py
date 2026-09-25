"""Register allocation with call-clobber constraints (public package).

Public surface::

    allocate(program, K) -> Allocation
    Program / Block / Instr dataclasses (Program accepts per-block
    non-negative integer ``frequencies``; omit them to keep the original
    loop-depth-weighted spill policy)
    program_from_dict(...)
    exception hierarchy: RegAllocError, ProgramFormatError,
                          InfeasibleProgramError, BudgetExhaustedError
    cost accounting: AccessStats, VarSpillCost, spill_cost_table,
                     actual_access_stats
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
    SpillDecision,
    allocate,
)
from .liveness import LivenessReport
from .cost import (
    AccessStats,
    VarSpillCost,
    actual_access_stats,
    spill_cost_table,
)

__all__ = [
    "allocate",
    "Allocation",
    "SpillDecision",
    "LivenessReport",
    "Program",
    "Block",
    "Instr",
    "program_from_dict",
    "RegAllocError",
    "ProgramFormatError",
    "InfeasibleProgramError",
    "BudgetExhaustedError",
    "AccessStats",
    "VarSpillCost",
    "actual_access_stats",
    "spill_cost_table",
]
