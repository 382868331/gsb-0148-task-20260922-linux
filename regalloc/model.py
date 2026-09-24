"""Public input data model and strict structural validation.

A program is a three-address CFG::

    Program(params, blocks)
    Block(label, instrs, succs, phis)
    PhiNode(def_var, sources=((pred_block_index, src_var), ...))
    Instr(uid, uses, defs, fixed={var: reg}, clobbers=(reg,...), note)

All variable and register numbers are non-negative ints.  Register numbers
are validated against K in :func:`regalloc.allocate` (the same program may be
allocated for different K).

Validation is strict and locatable: every error carries an error path such as
``blocks[3].instrs[2].fixed``.  ``bool`` is rejected wherever an int is
required and float NaN / +/-inf are rejected wherever a finite number is
required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- input interface bounds (contract) ------------------------------------

# Format version 1 (the original numeric-register contract).
FORMAT_V1 = 1
# Format version 2: target descriptions, width-aliased views, register
# classes/pairs, view clobbers, early-clobber defs, exception edges and
# explicit per-edge parallel copies.
FORMAT_V2 = 2
SUPPORTED_VERSIONS = (FORMAT_V1, FORMAT_V2)

MAX_BLOCKS = 40                    # v1 contract (kept for compatibility)
MAX_BLOCKS_V2 = 1_024              # v2 contract (the task scale needs >=100)
MAX_INSTRS_PER_BLOCK = 10_000      # v1
MAX_INSTRS_PER_BLOCK_V2 = 50_000   # v2 (the task scale needs >=1000 total)
MIN_K = 2
MAX_K = 8
MAX_BANKS_V2 = 32


class RegAllocError(Exception):
    """Base class for every error raised by this package."""


class ProgramFormatError(RegAllocError, ValueError):
    """The program / argument violates the structural contract."""


class InfeasibleProgramError(RegAllocError):
    """Hard constraints are contradictory; no allocation exists.

    This means *proven infeasible* (e.g. one variable pinned to two different
    registers, or a value that must survive a call in a register that call
    clobbers).  Distinct from :class:`BudgetExhaustedError`.
    """


class BudgetExhaustedError(RegAllocError):
    """The coloring search spent its budget without proving anything.

    Distinct from :class:`InfeasibleProgramError`: feasibility here is
    *unknown*, not disproven.
    """


def _err(path: str, msg: str) -> ProgramFormatError:
    return ProgramFormatError(f"{path}: {msg}" if path else msg)


def check_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as int; reject bool, float and out-of-range values.

    ``bool`` is rejected explicitly because ``isinstance(True, int)`` is true
    in Python; a JSON boolean is not a valid identifier/register here.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise _err(path, f"expected integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise _err(path, f"expected integer >= {minimum}, got {value}")
    return value


def check_finite_float(value: Any, path: str) -> float:
    """Reject non-finite floats (NaN / +inf / -inf).

    The core data model is integer-only, but entry points that accept numeric
    options go through this so the documented numeric contract holds
    everywhere.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _err(path, f"expected finite number, got {type(value).__name__}")
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        raise _err(path, "number must be finite (NaN and Infinity rejected)")
    return f


def _check_id_tuple(value: Any, path: str, *, unique: bool = True) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise _err(path, f"expected list/tuple, got {type(value).__name__}")
    out: list[int] = []
    for i, item in enumerate(value):
        out.append(check_int(item, f"{path}[{i}]", minimum=0))
    if unique and len(out) != len(set(out)):
        raise _err(path, "duplicate entries are not allowed")
    return tuple(out)


@dataclass(frozen=True)
class Instr:
    """One three-address instruction.

    ``fixed`` pins a use/def variable to a concrete physical register at THIS
    instruction point only (format v1: numeric register).  The allocator
    materialises the pin with short copies around the instruction (the
    variable's home register may differ); two operands pinned to the same
    register at one instruction is a proven infeasibility.  ``clobbers``
    lists registers that a call-like instruction destroys.

    Format-v2-only fields (rejected on v1 programs when non-empty):

    * ``fixedv`` — like ``fixed`` but names a target *view*;
    * ``classes`` — per-operand register class name ``{var: class}``
      (operands not listed use the target default class);
    * ``clobbersv`` — destroyed *views* (v2 replaces numeric ``clobbers``);
    * ``early_clobber`` — def vars written BEFORE the inputs are consumed;
    * ``except_to`` — successor block indices reachable by an exceptional
      transfer from THIS instruction (the defs are not written on that path).
    """

    uid: int
    uses: tuple[int, ...] = ()
    defs: tuple[int, ...] = ()
    fixed: dict[int, int] = field(default_factory=dict)
    clobbers: tuple[int, ...] = ()
    note: str = ""
    fixedv: dict[int, str] = field(default_factory=dict)
    classes: dict[int, str] = field(default_factory=dict)
    clobbersv: tuple[str, ...] = ()
    early_clobber: tuple[int, ...] = ()
    except_to: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        check_int(self.uid, "uid", minimum=0)
        if not isinstance(self.note, str):
            raise _err("note", "note must be a string")
        uses = _check_id_tuple(self.uses, "uses")
        defs = _check_id_tuple(self.defs, "defs")
        overlap = set(uses) & set(defs)
        if overlap:
            raise _err("", f"variable(s) both used and defined: {sorted(overlap)}")
        raw_fixed = self.fixed
        if not isinstance(raw_fixed, dict):
            raise _err("fixed", "fixed must be a dict {var: reg}")
        fixed: dict[int, int] = {}
        for key, reg in raw_fixed.items():
            # JSON keys may arrive as strings: accept "3" as well as 3.
            if isinstance(key, bool):
                raise _err("fixed", "boolean variable id is not allowed")
            if isinstance(key, str) and key.lstrip("-").isdigit():
                v = int(key)
            elif isinstance(key, int) and not isinstance(key, bool):
                v = key
            else:
                raise _err("fixed", f"bad variable id key: {key!r}")
            check_int(v, "fixed(var)", minimum=0)
            check_int(reg, f"fixed[{v}]", minimum=0)
            if v not in set(uses) | set(defs):
                raise _err(f"fixed[{v}]", "fixed variable must be used or defined here")
            fixed[v] = reg
        clobbers = _check_id_tuple(self.clobbers, "clobbers")

        # ---- v2-only fields (shape checks; semantic checks need the target
        # and happen in Program.__post_init__ / allocate) ----
        raw_fixedv = self.fixedv
        if not isinstance(raw_fixedv, dict):
            raise _err("fixedv", "fixedv must be a dict {var: view}")
        fixedv: dict[int, str] = {}
        for key, view in raw_fixedv.items():
            if isinstance(key, bool):
                raise _err("fixedv", "boolean variable id is not allowed")
            if isinstance(key, str) and key.lstrip("-").isdigit():
                v = int(key)
            elif isinstance(key, int) and not isinstance(key, bool):
                v = key
            else:
                raise _err("fixedv", f"bad variable id key: {key!r}")
            check_int(v, "fixedv(var)", minimum=0)
            if not isinstance(view, str) or not view:
                raise _err(f"fixedv[{v}]", "view name must be a non-empty string")
            if v not in set(uses) | set(defs):
                raise _err(f"fixedv[{v}]", "fixedv variable must be used or defined here")
            fixedv[v] = view
        if fixed and fixedv:
            raise _err("fixedv", "an instruction cannot use numeric fixed and view fixedv together")
        raw_classes = self.classes
        if not isinstance(raw_classes, dict):
            raise _err("classes", "classes must be a dict {var: class_name}")
        classes: dict[int, str] = {}
        for key, cname in raw_classes.items():
            if isinstance(key, bool):
                raise _err("classes", "boolean variable id is not allowed")
            if isinstance(key, str) and key.lstrip("-").isdigit():
                v = int(key)
            elif isinstance(key, int) and not isinstance(key, bool):
                v = key
            else:
                raise _err("classes", f"bad variable id key: {key!r}")
            check_int(v, "classes(var)", minimum=0)
            if not isinstance(cname, str) or not cname:
                raise _err(f"classes[{v}]", "class name must be a non-empty string")
            if v not in set(uses) | set(defs):
                raise _err(f"classes[{v}]", "class variable must be an operand here")
            classes[v] = cname
        if not isinstance(self.clobbersv, (list, tuple)) or not all(
            isinstance(x, str) and x for x in self.clobbersv
        ):
            raise _err("clobbersv", "clobbersv must be a list/tuple of non-empty view names")
        clobbersv = tuple(self.clobbersv)
        if len(set(clobbersv)) != len(clobbersv):
            raise _err("clobbersv", "duplicate clobbered views")
        early = _check_id_tuple(self.early_clobber, "early_clobber")
        for v in early:
            if v not in set(defs):
                raise _err(f"early_clobber[{v}]", "early-clobber variable must be a def here")
        exc = _check_id_tuple(self.except_to, "except_to")

        object.__setattr__(self, "uses", uses)
        object.__setattr__(self, "defs", defs)
        object.__setattr__(self, "fixed", fixed)
        object.__setattr__(self, "clobbers", clobbers)
        object.__setattr__(self, "fixedv", fixedv)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "clobbersv", clobbersv)
        object.__setattr__(self, "early_clobber", early)
        object.__setattr__(self, "except_to", exc)


@dataclass(frozen=True)
class PhiNode:
    """Phi definition with sources explicitly bound to predecessor edges.

    ``sources`` maps predecessor block index -> source variable on that edge.
    There must be exactly one entry per predecessor of the owning block.
    """

    def_var: int
    sources: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        check_int(self.def_var, "def", minimum=0)
        if not isinstance(self.sources, (list, tuple)):
            raise _err("sources", "expected list/tuple of (pred, src) pairs")
        norm: list[tuple[int, int]] = []
        seen_preds: set[int] = set()
        for i, pair in enumerate(self.sources):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise _err(f"sources[{i}]", "expected (pred_block, src_var) pair")
            pred = check_int(pair[0], f"sources[{i}].pred", minimum=0)
            src = check_int(pair[1], f"sources[{i}].sources[{i}].src", minimum=0)
            if pred in seen_preds:
                raise _err(f"sources[{i}]", f"duplicate source for predecessor {pred}")
            seen_preds.add(pred)
            norm.append((pred, src))
        object.__setattr__(self, "sources", tuple(norm))


@dataclass(frozen=True)
class EdgeCopies:
    """An explicit parallel copy on one CFG edge ``src_block -> dst_block``.

    ``pairs`` execute *simultaneously*: each item is
    ``(dst_var, src_var)``; the source values are all read before any
    destination is written.  Cycles are legal (e.g. a swap) and must be
    broken by the allocator using genuinely free alias locations or stack
    slots — never a location still carrying a live aliased value.
    """

    src_block: int
    dst_block: int
    pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        check_int(self.src_block, "edge_copies.src_block", minimum=0)
        check_int(self.dst_block, "edge_copies.dst_block", minimum=0)
        if not isinstance(self.pairs, (list, tuple)):
            raise _err("edge_copies.pairs", "expected list/tuple of [dst, src] pairs")
        norm: list[tuple[int, int]] = []
        dsts: set[int] = set()
        for i, pr in enumerate(self.pairs):
            if not isinstance(pr, (list, tuple)) or len(pr) != 2:
                raise _err(f"edge_copies[{self.src_block}->{self.dst_block}].pairs[{i}]",
                           "expected [dst, src] pair")
            d = check_int(pr[0], f"edge_copies.pairs[{i}].dst", minimum=0)
            s = check_int(pr[1], f"edge_copies.pairs[{i}].src", minimum=0)
            if d in dsts:
                raise _err(f"edge_copies[{self.src_block}->{self.dst_block}].pairs[{i}]",
                           f"destination v{d} written twice in one parallel copy")
            dsts.add(d)
            norm.append((d, s))
        object.__setattr__(self, "pairs", tuple(norm))


@dataclass(frozen=True)
class Block:
    label: str
    instrs: tuple[Instr, ...]
    succs: tuple[int, ...] = ()
    phis: tuple[PhiNode, ...] = ()
    edge_copies: tuple[EdgeCopies, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise _err("label", "label must be a non-empty string")
        if not isinstance(self.instrs, (list, tuple)):
            raise _err(f"blocks[{self.label!r}].instrs", "expected list/tuple")
        if len(self.instrs) > MAX_INSTRS_PER_BLOCK_V2:
            raise _err(
                f"blocks[{self.label!r}].instrs",
                f"block has {len(self.instrs)} instrs, limit {MAX_INSTRS_PER_BLOCK_V2}",
            )
        norm_instrs: list[Instr] = []
        for i, ins in enumerate(self.instrs):
            if not isinstance(ins, Instr):
                raise _err(f"blocks[{self.label!r}].instrs[{i}]", "expected Instr")
            norm_instrs.append(ins)
        succs = _check_id_tuple(self.succs, f"blocks[{self.label!r}].succs")
        phis = self.phis
        if not isinstance(phis, (list, tuple)):
            raise _err(f"blocks[{self.label!r}].phis", "expected list/tuple")
        norm_phis: list[PhiNode] = []
        for i, phi in enumerate(phis):
            if not isinstance(phi, PhiNode):
                raise _err(f"blocks[{self.label!r}].phis[{i}]", "expected PhiNode")
            norm_phis.append(phi)
        if len({phi.def_var for phi in norm_phis}) != len(norm_phis):
            raise _err(f"blocks[{self.label!r}].phis", "two phi nodes define the same variable")
        if not isinstance(self.edge_copies, (list, tuple)):
            raise _err(f"blocks[{self.label!r}].edge_copies", "expected list/tuple")
        norm_ec: list[EdgeCopies] = []
        for i, ec in enumerate(self.edge_copies):
            if not isinstance(ec, EdgeCopies):
                raise _err(f"blocks[{self.label!r}].edge_copies[{i}]", "expected EdgeCopies")
            norm_ec.append(ec)
        object.__setattr__(self, "instrs", tuple(norm_instrs))
        object.__setattr__(self, "succs", succs)
        object.__setattr__(self, "phis", tuple(norm_phis))
        object.__setattr__(self, "edge_copies", tuple(norm_ec))


@dataclass(frozen=True)
class StackSpec:
    """Stack layout contract (v2).

    * ``align`` — every emitted spill slot base is aligned to this many slot
      units (power of two, >=1).  Double-width values request 2-unit slots;
      the slot allocator packs single-unit slots and skips padding so that
      each live range gets an aligned, non-overlapping interval.
    * ``min_slots`` — the frame always provides at least this many units
      (useful for fixed ABI layouts); default 0.
    """

    align: int = 1
    min_slots: int = 0

    def __post_init__(self) -> None:
        check_int(self.align, "stack.align", minimum=1)
        if self.align & (self.align - 1):
            raise _err("stack.align", f"alignment must be a power of two, got {self.align}")
        check_int(self.min_slots, "stack.min_slots", minimum=0)


@dataclass(frozen=True)
class Program:
    blocks: tuple[Block, ...]
    params: tuple[int, ...] = ()
    name: str = ""
    version: int = FORMAT_V1
    target: object | None = None     # TargetDesc when version == 2
    stack: StackSpec | None = None
    # v2-only: per-parameter register class name {param_var: class_name};
    # absent parameters use the target default class.
    param_classes: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, (list, tuple)) or not self.blocks:
            raise ProgramFormatError("blocks: expected a non-empty list/tuple")
        if self.version not in SUPPORTED_VERSIONS:
            raise ProgramFormatError(
                f"version: unsupported format version {self.version!r}, "
                f"supported: {SUPPORTED_VERSIONS}"
            )
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise ProgramFormatError("version: expected an integer")
        max_blocks = MAX_BLOCKS if self.version == FORMAT_V1 else MAX_BLOCKS_V2
        if len(self.blocks) > max_blocks:
            raise ProgramFormatError(
                f"blocks: program has {len(self.blocks)} blocks, limit {max_blocks}"
            )
        blocks = tuple(self.blocks)
        for i, b in enumerate(blocks):
            if not isinstance(b, Block):
                raise _err(f"blocks[{i}]", "expected Block")
        params = _check_id_tuple(self.params or (), "params")
        if not isinstance(self.name, str):
            raise _err("name", "name must be a string")
        if self.stack is not None and not isinstance(self.stack, StackSpec):
            raise _err("stack", "expected a StackSpec instance")
        if not isinstance(self.param_classes, dict):
            raise _err("param_classes", "expected a dict {param: class_name}")

        # v1/v2 field gating: v2-only instruction fields are rejected on v1
        # programs so the original format's meaning is unchanged.
        for i, b in enumerate(blocks):
            for j, ins in enumerate(b.instrs):
                if self.version == FORMAT_V1:
                    if ins.fixedv or ins.classes or ins.clobbersv or ins.early_clobber or ins.except_to:
                        raise _err(
                            f"blocks[{i}].instrs[{j}]",
                            "fixedv/classes/clobbersv/early_clobber/except_to require "
                            "version=2 and a target description",
                        )
                if ins.fixed and self.version == FORMAT_V2:
                    raise _err(
                        f"blocks[{i}].instrs[{j}].fixed",
                        "numeric fixed is a v1 field; use fixedv with view names in v2",
                    )
                if ins.clobbers and self.version == FORMAT_V2:
                    raise _err(
                        f"blocks[{i}].instrs[{j}].clobbers",
                        "numeric clobbers is a v1 field; use clobbersv with view names in v2",
                    )
            if self.version == FORMAT_V1 and b.edge_copies:
                raise _err(f"blocks[{i}].edge_copies", "edge_copies requires version=2")

        target = self.target
        if self.version == FORMAT_V2:
            from .target import TargetDesc
            if target is None:
                raise ProgramFormatError("version=2 programs require a target description")
            if not isinstance(target, TargetDesc):
                raise _err("target", "expected a TargetDesc instance")
            target.resolved()  # full alias/class/callee-save validation
            if self.stack is None:
                object.__setattr__(self, "stack", StackSpec())
        elif target is not None:
            raise ProgramFormatError("a target description requires version=2")

        n = len(blocks)
        for i, b in enumerate(blocks):
            for s in b.succs:
                if s >= n:
                    raise _err(f"blocks[{i}].succs", f"successor {s} out of range (0..{n - 1})")

        preds: list[set[int]] = [set() for _ in range(n)]
        for i, b in enumerate(blocks):
            for s in b.succs:
                preds[s].add(i)

        # exception edges: an instruction may transfer control to a block;
        # the target must be a (non-fallthrough) successor — checked after
        # preds are known; targets need not be listed as a block succ here
        # because they ARE control flow, so auto-presence is validated.
        for i, b in enumerate(blocks):
            for j, ins in enumerate(b.instrs):
                for t in ins.except_to:
                    if t >= n:
                        raise _err(
                            f"blocks[{i}].instrs[{j}].except_to",
                            f"exception target {t} out of range",
                        )
                    if t not in b.succs:
                        raise _err(
                            f"blocks[{i}].instrs[{j}].except_to",
                            f"exception target block {t} must also be a successor of block {i}",
                        )
                    if t == i:
                        raise _err(
                            f"blocks[{i}].instrs[{j}].except_to",
                            "an exception edge cannot target its own block",
                        )

        # edge copies must sit on real edges; at most one group per edge
        seen_edges: set[tuple[int, int]] = set()
        for i, b in enumerate(blocks):
            for ec in b.edge_copies:
                if ec.src_block != i:
                    raise _err(
                        f"blocks[{i}].edge_copies",
                        f"edge copy declares src_block {ec.src_block}, must be {i}",
                    )
                if ec.dst_block not in b.succs:
                    raise _err(
                        f"blocks[{i}].edge_copies",
                        f"edge copy to block {ec.dst_block} is not a successor of block {i}",
                    )
                key = (ec.src_block, ec.dst_block)
                if key in seen_edges:
                    raise _err(f"blocks[{i}].edge_copies", f"duplicate parallel copy on edge {key[0]}->{key[1]}")
                seen_edges.add(key)

        # reachability from entry block 0
        seen = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            for s in blocks[cur].succs:
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        unreachable = sorted(set(range(n)) - seen)
        if unreachable:
            raise _err("blocks", f"unreachable block(s): {unreachable}")

        # phi sources must match the predecessor set exactly
        for i, b in enumerate(blocks):
            if b.phis and i == 0 and not preds[0]:
                raise _err("blocks[0].phis", "the entry block cannot contain phi nodes (no incoming edge can supply a source)")
            for phi in b.phis:
                pred_set = set(preds[i])
                phi_preds = {p for p, _ in phi.sources}
                missing = sorted(pred_set - phi_preds)
                extra = sorted(phi_preds - pred_set)
                if missing:
                    raise _err(
                        f"blocks[{i}].phis",
                        f"phi for v{phi.def_var} missing sources on edges from {missing}",
                    )
                if extra:
                    raise _err(
                        f"blocks[{i}].phis",
                        f"phi for v{phi.def_var} names non-predecessor blocks {extra}",
                    )

        # every referenced variable must be a parameter or defined somewhere
        defined: set[int] = set(params)
        referenced: set[int] = set()
        for b in blocks:
            for ins in b.instrs:
                defined.update(ins.defs)
                referenced.update(ins.uses)
                referenced.update(ins.fixed)
                referenced.update(ins.fixedv)
                referenced.update(ins.classes)
            for ec in b.edge_copies:
                for d, s in ec.pairs:
                    referenced.add(s)
                    defined.add(d)  # edge-copy destinations are defined by the copy
            for phi in b.phis:
                defined.add(phi.def_var)
                referenced.add(phi.def_var)
                for _, src in phi.sources:
                    referenced.add(src)
        unknown = sorted(referenced - defined)
        if unknown:
            raise _err("blocks", f"variable(s) never defined and not parameters: {unknown}")
        phi_defs = {phi.def_var for b in blocks for phi in b.phis}
        if set(params) & phi_defs:
            raise _err("params", "a parameter cannot also be a phi definition")
        param_defs = set(params) & {d for b in blocks for ins in b.instrs for d in ins.defs}
        if param_defs:
            raise _err("params", f"parameter(s) also defined by an instruction: {sorted(param_defs)}")
        # an edge-copy destination must not be defined elsewhere (it is born
        # on the edge, exactly like a phi result is born at block entry)
        edge_dst_vars = {d for b in blocks for ec in b.edge_copies for d, _ in ec.pairs}
        other_defs = {d for b in blocks for ins in b.instrs for d in ins.defs} | phi_defs
        clash = sorted(edge_dst_vars & (other_defs | set(params)))
        if clash:
            raise _err("blocks", f"edge-copy destination(s) also defined elsewhere: {clash}")

        # unique instruction uids
        uids: set[int] = set()
        for i, b in enumerate(blocks):
            for ins in b.instrs:
                if ins.uid in uids:
                    raise _err(f"blocks[{i}].instrs", f"duplicate instruction uid {ins.uid}")
                uids.add(ins.uid)

        # ---- v2 target semantic validation ----
        if self.version == FORMAT_V2:
            resolved = target.resolved()
            if resolved.num_banks > MAX_BANKS_V2:
                raise _err("target.views", f"too many banks: {resolved.num_banks} > {MAX_BANKS_V2}")
            for pv, pc in self.param_classes.items():
                if isinstance(pv, str) and pv.lstrip("-").isdigit():
                    pass
                if not isinstance(pc, str) or not resolved.has_class(pc):
                    raise _err(f"param_classes[{pv}]", f"unknown register class {pc!r}")
            norm_pc: dict[int, str] = {}
            for key, cname in self.param_classes.items():
                if isinstance(key, bool):
                    raise _err("param_classes", "boolean variable id is not allowed")
                v = int(key) if isinstance(key, str) and key.lstrip("-").isdigit() else key
                if not isinstance(v, int) or v not in set(params):
                    raise _err(f"param_classes[{key}]", "class given for a non-parameter variable")
                norm_pc[v] = cname
            object.__setattr__(self, "param_classes", norm_pc)
            for i, b in enumerate(blocks):
                for j, ins in enumerate(b.instrs):
                    operands = set(ins.uses) | set(ins.defs)
                    for v, view in ins.fixedv.items():
                        if not resolved.has_view(view):
                            raise _err(f"blocks[{i}].instrs[{j}].fixedv[{v}]",
                                       f"unknown view {view!r}")
                    for v, cname in ins.classes.items():
                        if not resolved.has_class(cname):
                            raise _err(f"blocks[{i}].instrs[{j}].classes[{v}]",
                                       f"unknown register class {cname!r}")
                    for view in ins.clobbersv:
                        if not resolved.has_view(view):
                            raise _err(f"blocks[{i}].instrs[{j}].clobbersv",
                                       f"unknown view {view!r}")
                    for v in operands:
                        if v in ins.fixedv:
                            view = ins.fixedv[v]
                            cname = ins.classes.get(v, resolved.default_class_name)
                            info = resolved.class_info(cname)
                            if resolved.view(view).bank not in (
                                bnk for opt in info.options for bnk in opt
                            ):
                                raise _err(
                                    f"blocks[{i}].instrs[{j}].fixedv[{v}]",
                                    f"view {view!r} is not an option of class {cname!r}",
                                )

        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "params", params)

    def var_count(self) -> int:
        """One past the largest variable id used (ids may be sparse)."""
        m = -1
        for v in self.params:
            m = max(m, v)
        for b in self.blocks:
            for ins in b.instrs:
                m = max(m, max(ins.uses, default=-1), max(ins.defs, default=-1),
                        max(ins.fixed, default=-1), max(ins.fixedv, default=-1))
            for phi in b.phis:
                m = max(m, phi.def_var)
                for _, src in phi.sources:
                    m = max(m, src)
            for ec in b.edge_copies:
                for d, s in ec.pairs:
                    m = max(m, d, s)
        return m + 1

    def is_v2(self) -> bool:
        return self.version == FORMAT_V2


# ---- dict (e.g. JSON) construction ----------------------------------------

def _at(prefix: str, i: int) -> str:
    return f"{prefix}[{i}]"


def program_from_dict(data: Any) -> Program:
    """Build a :class:`Program` from plain dicts/lists.

    v1 shape::

        {"name": str?, "params": [int]?,
         "blocks": [{"label": str?,
                     "instrs": [{"uid": int, "uses": [int], "defs": [int],
                                 "fixed": {"0": 1}, "clobbers": [int],
                                 "note": str?}],
                     "succs": [int],
                     "phis": [{"def": int, "sources": [[pred, src], ...]}]}]}

    v2 adds (top level): ``"version": 2``, ``"target"`` (a target
    description mapping, see :func:`regalloc.target.target_from_dict`),
    ``"stack": {"align": int, "min_slots": int}``,
    ``"param_classes": {"<var>": class_name}``; instructions may carry
    ``fixedv``, ``classes``, ``clobbersv``, ``early_clobber``, ``except_to``;
    blocks may carry ``edge_copies``:
    ``{"src": int, "dst": int, "pairs": [[dst_var, src_var], ...]}``.
    """
    if not isinstance(data, dict):
        raise _err("program", "expected a mapping/object")
    version = data.get("version", FORMAT_V1)
    blocks_raw = data.get("blocks")
    if not isinstance(blocks_raw, list) or not blocks_raw:
        raise _err("blocks", "expected a non-empty list")
    blocks: list[Block] = []
    for bi, bd in enumerate(blocks_raw):
        bp = f"blocks[{bi}]"
        if not isinstance(bd, dict):
            raise _err(bp, "expected an object")
        label = bd.get("label", f"b{bi}")
        if not isinstance(label, str):
            raise _err(f"{bp}.label", "label must be a string")
        instrs: list[Instr] = []
        for ii, idata in enumerate(bd.get("instrs", [])):
            ip = f"{bp}.instrs[{ii}]"
            if not isinstance(idata, dict):
                raise _err(ip, "expected an object")
            if "uid" not in idata:
                raise _err(f"{ip}.uid", "uid is required")
            kwargs = dict(
                uid=idata["uid"],
                uses=tuple(idata.get("uses", ())),
                defs=tuple(idata.get("defs", ())),
                fixed=dict(idata.get("fixed", {})),
                clobbers=tuple(idata.get("clobbers", ())),
                note=str(idata.get("note", "")),
            )
            if version == FORMAT_V2:
                kwargs.update(
                    fixedv=dict(idata.get("fixedv", {})),
                    classes=dict(idata.get("classes", {})),
                    clobbersv=tuple(idata.get("clobbersv", ())),
                    early_clobber=tuple(idata.get("early_clobber", ())),
                    except_to=tuple(idata.get("except_to", ())),
                )
            try:
                instrs.append(Instr(**kwargs))
            except ProgramFormatError as exc:
                raise ProgramFormatError(f"{ip}: {exc}") from None
        phis: list[PhiNode] = []
        for pi, pdata in enumerate(bd.get("phis", [])):
            pp = f"{bp}.phis[{pi}]"
            if not isinstance(pdata, dict):
                raise _err(pp, "expected an object")
            if "def" not in pdata:
                raise _err(f"{pp}.def", "phi def is required")
            srcs = []
            for si, pair in enumerate(pdata.get("sources", [])):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise _err(f"{pp}.sources[{si}]", "expected [pred, src] pair")
                srcs.append((pair[0], pair[1]))
            try:
                phis.append(
                    PhiNode(
                        def_var=pdata["def"],
                        sources=tuple(srcs),
                    )
                )
            except ProgramFormatError as exc:
                raise ProgramFormatError(f"{pp}: {exc}") from None
        edge_copies: list[EdgeCopies] = []
        for ei, ecdata in enumerate(bd.get("edge_copies", ())):
            ep = f"{bp}.edge_copies[{ei}]"
            if not isinstance(ecdata, dict):
                raise _err(ep, "expected an object")
            if "dst" not in ecdata:
                raise _err(f"{ep}.dst", "edge copy dst is required")
            pairs = []
            for pair in ecdata.get("pairs", ()):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise _err(f"{ep}.pairs", "expected [dst_var, src_var]")
                pairs.append((pair[0], pair[1]))
            edge_copies.append(EdgeCopies(
                src_block=ecdata.get("src", bi),
                dst_block=ecdata["dst"],
                pairs=tuple(pairs),
            ))
        blocks.append(
            Block(
                label=label,
                instrs=tuple(instrs),
                succs=tuple(bd.get("succs", ())),
                phis=tuple(phis),
                edge_copies=tuple(edge_copies),
            )
        )
    params = tuple(data.get("params", ()))
    target = None
    stack = None
    param_classes = {}
    if version == FORMAT_V2:
        from .target import target_from_dict
        if "target" not in data:
            raise _err("target", "version=2 program requires a target description")
        target = target_from_dict(data["target"])
        if "stack" in data and data["stack"] is not None:
            sd = data["stack"]
            if not isinstance(sd, dict):
                raise _err("stack", "expected an object")
            stack = StackSpec(
                align=sd.get("align", 1),
                min_slots=sd.get("min_slots", 0),
            )
        param_classes = dict(data.get("param_classes", {}))
    try:
        return Program(
            blocks=tuple(blocks), params=params, name=str(data.get("name", "")),
            version=version, target=target, stack=stack,
            param_classes=param_classes,
        )
    except ProgramFormatError as exc:
        # make instr-level paths locatable when constructed from raw blocks
        raise exc
