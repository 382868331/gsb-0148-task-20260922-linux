"""Register allocation with call-clobber constraints (public package).

V1 public surface (unchanged)::

    allocate(program, K) -> Allocation
    Program / Block / Instr dataclasses
    program_from_dict(...)
    exception hierarchy: RegAllocError, ProgramFormatError,
                          InfeasibleProgramError, BudgetExhaustedError

V2 public surface (register views / exceptional edges / stack slots)::

    Machine / RegView / machine_from_dict / flat_machine
    allocate2(program, machine) -> AllocationResult   (status object)
    verify_allocation(program, machine, result) -> VerifyReport
    symbolic2.equivalence_check2(program, machine, result)
    compat.allocate_legacy_via_v2(program, K)         (migration adapter)
"""

from .model import (
    Block,
    Instr,
    PhiNode,
    Program,
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
from .liveness import LivenessReport
from .machine import (
    Machine,
    RegView,
    flat_machine,
    machine_from_dict,
)
from .allocator2 import (
    AllocationResult,
    Slot,
    allocate2,
)
from .verify import (
    VerifyReport,
    verify_allocation,
)

__all__ = [
    # v1
    "allocate",
    "Allocation",
    "LivenessReport",
    "Program",
    "Block",
    "Instr",
    "PhiNode",
    "program_from_dict",
    "RegAllocError",
    "ProgramFormatError",
    "InfeasibleProgramError",
    "BudgetExhaustedError",
    # v2
    "Machine",
    "RegView",
    "flat_machine",
    "machine_from_dict",
    "allocate2",
    "AllocationResult",
    "Slot",
    "verify_allocation",
    "VerifyReport",
]
