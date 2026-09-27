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

MAX_BLOCKS = 40
MAX_INSTRS_PER_BLOCK = 10_000
MIN_K = 2
MAX_K = 8


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
    instruction point only.  The allocator materialises the pin with short
    copies around the instruction (the variable's home register may differ);
    two operands pinned to the same register at one instruction is a proven
    infeasibility.  ``clobbers`` lists registers that a call-like instruction
    destroys.
    """

    uid: int
    uses: tuple[int, ...] = ()
    defs: tuple[int, ...] = ()
    fixed: dict[int, int] = field(default_factory=dict)
    clobbers: tuple[int, ...] = ()
    note: str = ""

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
        object.__setattr__(self, "uses", uses)
        object.__setattr__(self, "defs", defs)
        object.__setattr__(self, "fixed", fixed)
        object.__setattr__(self, "clobbers", clobbers)


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
class Block:
    label: str
    instrs: tuple[Instr, ...]
    succs: tuple[int, ...] = ()
    phis: tuple[PhiNode, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise _err("label", "label must be a non-empty string")
        if not isinstance(self.instrs, (list, tuple)):
            raise _err(f"blocks[{self.label!r}].instrs", "expected list/tuple")
        if len(self.instrs) > MAX_INSTRS_PER_BLOCK:
            raise _err(
                f"blocks[{self.label!r}].instrs",
                f"block has {len(self.instrs)} instrs, limit {MAX_INSTRS_PER_BLOCK}",
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
        object.__setattr__(self, "instrs", tuple(norm_instrs))
        object.__setattr__(self, "succs", succs)
        object.__setattr__(self, "phis", tuple(norm_phis))


@dataclass(frozen=True)
class Program:
    blocks: tuple[Block, ...]
    params: tuple[int, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, (list, tuple)) or not self.blocks:
            raise ProgramFormatError("blocks: expected a non-empty list/tuple")
        if len(self.blocks) > MAX_BLOCKS:
            raise ProgramFormatError(
                f"blocks: program has {len(self.blocks)} blocks, limit {MAX_BLOCKS}"
            )
        blocks = tuple(self.blocks)
        for i, b in enumerate(blocks):
            if not isinstance(b, Block):
                raise _err(f"blocks[{i}]", "expected Block")
        params = _check_id_tuple(self.params or (), "params")
        if not isinstance(self.name, str):
            raise _err("name", "name must be a string")

        n = len(blocks)
        for i, b in enumerate(blocks):
            for s in b.succs:
                if s >= n:
                    raise _err(f"blocks[{i}].succs", f"successor {s} out of range (0..{n - 1})")

        preds: list[set[int]] = [set() for _ in range(n)]
        for i, b in enumerate(blocks):
            for s in b.succs:
                preds[s].add(i)

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

        # unique instruction uids
        uids: set[int] = set()
        for i, b in enumerate(blocks):
            for ins in b.instrs:
                if ins.uid in uids:
                    raise _err(f"blocks[{i}].instrs", f"duplicate instruction uid {ins.uid}")
                uids.add(ins.uid)

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
                        max(ins.fixed, default=-1))
            for phi in b.phis:
                m = max(m, phi.def_var)
                for _, src in phi.sources:
                    m = max(m, src)
        return m + 1


# ---- rematerialisation policy ---------------------------------------------

@dataclass(frozen=True)
class RematSpec:
    """Caller-supplied policy: which pure instructions may be recomputed.

    ``costs`` maps an instruction ``uid`` to a non-negative integer
    rematerialisation cost, comparable to one load/store round trip; 0
    marks a value literally computable from constants with no operand
    reads (e.g. a load-immediate).  Marking an instruction is a
    *permission*: the allocator recomputes its result(s) near a use
    instead of keeping the value (or spilling it) only when that relieves
    register pressure.

    Only *pure* instructions may be marked: no ``clobbers`` (call-like
    instructions destroy state), no ``fixed`` operand requirements, and
    at least one def.  A phi result is never rematerialisable (it has no
    single definition site).  The caller makes the purity promise; only
    structural properties are checked here.
    """

    costs: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.costs, dict):
            raise _err("remat", "expected a mapping {uid: cost}")
        costs: dict[int, int] = {}
        for key, value in self.costs.items():
            if isinstance(key, bool):
                raise _err("remat", "boolean instruction uid is not allowed")
            if isinstance(key, str) and key.lstrip("-").isdigit():
                uid = int(key)
            elif isinstance(key, int) and not isinstance(key, bool):
                uid = key
            else:
                raise _err("remat", f"bad instruction uid key: {key!r}")
            check_int(uid, "remat(uid)", minimum=0)
            check_int(value, f"remat[{uid}]", minimum=0)
            costs[uid] = value
        object.__setattr__(self, "costs", costs)

    @classmethod
    def coerce(cls, obj: Any) -> "RematSpec":
        """Accept a :class:`RematSpec` or a plain ``{uid: cost}`` dict."""
        if isinstance(obj, RematSpec):
            return obj
        if obj is None:
            return cls()
        if isinstance(obj, dict):
            return cls(costs=obj)
        raise _err("remat", "expected a RematSpec or {uid: cost} dict")


def validate_remat_spec(program: Program, spec: RematSpec) -> None:
    """Cross-check a rematerialisation policy against a concrete program.

    Every marked uid must exist, define at least one variable, be free of
    call clobbers and fixed-register requirements.  Errors carry
    locatable paths; the function is side-effect free.
    """
    by_uid: dict[int, tuple[int, Instr]] = {}
    for bi, block in enumerate(program.blocks):
        for ins in block.instrs:
            by_uid[ins.uid] = (bi, ins)
    for uid in sorted(spec.costs):
        if uid not in by_uid:
            raise _err(f"remat[{uid}]", "no instruction with this uid")
        bi, ins = by_uid[uid]
        path = f"blocks[{bi}].instrs uid={uid}"
        if not ins.defs:
            raise _err(
                f"remat[{uid}]",
                f"{path} defines nothing; a recomputed value needs a def",
            )
        if ins.clobbers:
            raise _err(
                f"remat[{uid}]",
                f"{path} clobbers registers; only pure side-effect-free "
                f"operations may be rematerialised",
            )
        if ins.fixed:
            raise _err(
                f"remat[{uid}]",
                f"{path} has fixed-register operands; it cannot be freely "
                f"recomputed at arbitrary use points",
            )


# ---- dict (e.g. JSON) construction ----------------------------------------

def _at(prefix: str, i: int) -> str:
    return f"{prefix}[{i}]"


def program_from_dict(data: Any) -> Program:
    """Build a :class:`Program` from plain dicts/lists.

    Expected shape::

        {"name": str?, "params": [int]?,
         "blocks": [{"label": str?,
                     "instrs": [{"uid": int, "uses": [int], "defs": [int],
                                 "fixed": {"0": 1}, "clobbers": [int],
                                 "note": str?}],
                     "succs": [int],
                     "phis": [{"def": int, "sources": [[pred, src], ...]}]}]}
    """
    if not isinstance(data, dict):
        raise _err("program", "expected a mapping/object")
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
            try:
                instrs.append(
                    Instr(
                        uid=idata["uid"],
                        uses=tuple(idata.get("uses", ())),
                        defs=tuple(idata.get("defs", ())),
                        fixed=dict(idata.get("fixed", {})),
                        clobbers=tuple(idata.get("clobbers", ())),
                        note=str(idata.get("note", "")),
                    )
                )
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
        blocks.append(
            Block(
                label=label,
                instrs=tuple(instrs),
                succs=tuple(bd.get("succs", ())),
                phis=tuple(phis),
            )
        )
    params = tuple(data.get("params", ()))
    try:
        return Program(blocks=tuple(blocks), params=params, name=str(data.get("name", "")))
    except ProgramFormatError as exc:
        # make instr-level paths locatable when constructed from raw blocks
        raise exc
