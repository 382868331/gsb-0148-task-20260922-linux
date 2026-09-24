"""Register allocation with call-clobber constraints (public package).

Public surface::

    allocate(program)                  # format v2 (target description)
    allocate(program, K)               # format v1 (numeric registers)
    Program / Block / Instr dataclasses (both versions)
    program_from_dict(...)
    exception hierarchy: RegAllocError, ProgramFormatError,
                          InfeasibleProgramError, BudgetExhaustedError,
                          VerificationFailureError (v2)
    v2 target description: TargetDesc, RegView, RegClass, StackSpec,
                           default_target, tiny_target, target_from_dict
"""

from .model import (
    Block,
    EdgeCopies,
    Instr,
    PhiNode,
    Program,
    StackSpec,
    FORMAT_V1,
    FORMAT_V2,
    program_from_dict,
)
from .allocator import (
    Allocation,
    RegAllocError,
    ProgramFormatError,
    InfeasibleProgramError,
    BudgetExhaustedError,
    allocate,
)
from .allocator2 import (
    AllocationV2,
    VerificationFailureError,
    allocate_v2,
)
from .liveness import LivenessReport
from .liveness2 import LivenessReportV2
from .lower2 import AllocatedProgram2, MI2
from .target import (
    TargetDesc,
    RegView,
    RegClass,
    ClassInfo,
    default_target,
    tiny_target,
    target_from_dict,
)
from .checker2 import verify_allocation, VerificationReport

__all__ = [
    "allocate",
    "allocate_v2",
    "Allocation",
    "AllocationV2",
    "LivenessReport",
    "LivenessReportV2",
    "Program",
    "Block",
    "Instr",
    "PhiNode",
    "EdgeCopies",
    "StackSpec",
    "TargetDesc",
    "RegView",
    "RegClass",
    "ClassInfo",
    "AllocatedProgram2",
    "MI2",
    "default_target",
    "tiny_target",
    "target_from_dict",
    "program_from_dict",
    "verify_allocation",
    "VerificationReport",
    "FORMAT_V1",
    "FORMAT_V2",
    "RegAllocError",
    "ProgramFormatError",
    "InfeasibleProgramError",
    "BudgetExhaustedError",
    "VerificationFailureError",
]
