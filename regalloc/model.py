"""Public data structures and error types for the register allocator.

The input IR is a three-address control-flow graph:

* :class:`Program` contains ordered :class:`Block` objects and an entry id.
* Every :class:`Instruction` declares its ``uses`` / ``defs`` (virtual variable
  ids), optional fixed-register requirements for operands and results, and a
  ``clobbers`` tuple of physical registers destroyed by the instruction (the
  call-clobber set).
* :class:`Phi` nodes live at block entries; every phi source is explicitly
  bound to a predecessor block id (edge-sensitive SSA-style phis).

Physical registers are the integers ``0 .. K-1`` with ``K`` in ``2 .. 8``.
The allocator never mutates the input program.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Union


class AllocatorError(Exception):
    """Base class for every error raised by this package."""


class InvalidArgumentError(TypeError, AllocatorError):
    """A scalar API argument has the wrong type or value (e.g. ``K=True``)."""


class InvalidProgramError(ValueError, AllocatorError):
    """The Program/Block/Instruction IR violates the documented contract.

    The optional ``location`` string pinpoints the offending construct
    (block id, instruction index, ...) so that callers can locate it.
    """

    def __init__(self, message: str, location: Optional[str] = None) -> None:
        if location:
            message = f"{location}: {message}"
        super().__init__(message)
        self.location = location


class UnsatisfiableProgram(AllocatorError):
    """Allocation is *provably* impossible for this program and K.

    Raised for forced conflicts that spilling cannot resolve (two distinct
    simultaneously-live values both forced into the same physical register at
    the same instruction), distinct from search-budget exhaustion.
    """


class AllocationBudgetExhausted(AllocatorError):
    """The iterative spill/recolor search hit its iteration budget.

    This does **not** claim the program is unsatisfiable; it only says the
    allocator's bounded search gave up.
    """


@dataclass(frozen=True)
class Phi:
    """A phi node: ``dst = phi(pred_block_id -> src_variable, ...)``.

    ``srcs`` must contain exactly one entry per predecessor of the owning
    block. ``fixed_def`` optionally forces the destination to a physical
    register.
    """

    dst: str
    srcs: Mapping[str, str] = field(default_factory=dict)
    fixed_def: Optional[int] = None


@dataclass(frozen=True)
class Instruction:
    """A three-address instruction.

    * ``uses`` / ``defs``: virtual variable ids (defs normally has 0/1 item).
    * ``fixed_use`` / ``fixed_def``: variable id -> required physical register.
    * ``clobbers``: physical registers destroyed (call-clobber set). Values
      live across the instruction must not reside in any of them.
    """

    opcode: str
    uses: tuple[str, ...] = ()
    defs: tuple[str, ...] = ()
    fixed_use: Mapping[str, int] = field(default_factory=dict)
    fixed_def: Mapping[str, int] = field(default_factory=dict)
    clobbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class Block:
    id: str
    phis: tuple[Phi, ...] = ()
    instrs: tuple[Instruction, ...] = ()
    succs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Program:
    entry: str
    blocks: tuple[Block, ...]
    params: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Output structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Reg:
    """A concrete physical register location."""

    i: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"r{self.i}"


@dataclass(frozen=True)
class Slot:
    """A stack spill slot (word-sized, identity is the slot index)."""

    i: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[sp+{self.i}]"


Loc = Union[Reg, Slot]


@dataclass(frozen=True)
class MInstr:
    """A concrete machine-like instruction emitted by the allocator.

    Opcodes:
      * ``"param"``: entry materialization of one incoming parameter (write a
        Reg with that parameter's initial value); appears only at the head of
        the entry block, once per parameter.
      * ``"mov"``   : one read, one write (register copy).
      * ``"load"``  : read a Slot, write a Reg (reload).
      * ``"store"`` : read a Reg, write a Slot (spill).
      * anything else: the original user opcode; semantics are opaque to the
        allocator. ``origin`` is the index of that instruction inside its
        original block (None for allocator-inserted glue).
    For a ``call``-like instruction, ``clobbers`` lists physical registers
    whose contents are destroyed unless they are also in ``writes``.
    """

    opcode: str
    reads: tuple[Loc, ...]
    writes: tuple[Loc, ...]
    origin: Optional[int]
    clobbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class MBlock:
    id: str
    instrs: tuple[MInstr, ...]
    succs: tuple[str, ...]


@dataclass(frozen=True)
class AllocatedProgram:
    entry: str
    blocks: tuple[MBlock, ...]


@dataclass(frozen=True)
class Liveness:
    """Edge-sensitive liveness of the ORIGINAL program (sorted tuples).

    * ``edge_live``: variables live immediately on each CFG edge
      ``(pred_id, succ_id)``; phi sources are counted on their bound edge.
    * ``live_in`` / ``live_out``: per-block boundary sets. ``live_in``
      contains the values arriving from predecessors (phi destinations are
      born at the entry and are therefore not in ``live_in``).
    * ``points``: for ``(block_id, original instruction index)`` the
      ``(live_before, live_after)`` sets over non-phi instructions.
    """

    edge_live: Mapping[tuple[str, str], tuple[str, ...]]
    live_in: Mapping[str, tuple[str, ...]]
    live_out: Mapping[str, tuple[str, ...]]
    points: Mapping[tuple[str, int], tuple[tuple[str, ...], tuple[str, ...]]]


@dataclass(frozen=True)
class Allocation:
    program: AllocatedProgram
    #: virtual variable id -> physical register, or None if it was spilled.
    assignment: Mapping[str, Optional[int]]
    #: spilled variable id -> stack slot index (assigned by variable id order).
    stack_slots: Mapping[str, int]
    live: Liveness
    K: int
    iterations: int
    #: Extra stack slots used only to break parallel-copy cycles (usually
    #: empty; SSA phi lowering in this package does not produce cycles, but
    #: code generation keeps the safety mechanism explicit and verifiable).
    scratch_slots: tuple[int, ...] = ()
