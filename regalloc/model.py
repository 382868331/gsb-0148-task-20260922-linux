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

Format v2 (register views / exceptional edges)
----------------------------------------------
The v1 contract above is unchanged.  A program becomes a *v2 program* by
passing ``Program(var_classes={...})``; v2 programs are allocated with
:func:`regalloc.allocate2` against a :class:`regalloc.machine.Machine`.
V2 adds optional fields (all defaulting to the v1 behaviour):

* ``Instr.early_clobber``: defs written before the uses are read;
* ``Instr.may_throw`` / ``Block.exc_succs``: exceptional control flow;
* ``Instr.is_call`` / ``Instr.clobber_views``: machine-level call clobbers;
* ``Instr.fixed_views``: pin an operand to a named register view.

Using any v2 feature without ``var_classes`` is a format error, and the
legacy :func:`regalloc.allocate` entry point rejects v2 programs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- input interface bounds (contract) ------------------------------------

MAX_BLOCKS = 40
MAX_BLOCKS_V2 = 128
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


def _check_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _err(path, f"expected bool, got {type(value).__name__}")
    return value


def _check_view_name(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _err(path, "expected a non-empty view-name string")
    return value


def _check_view_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise _err(path, f"expected list/tuple, got {type(value).__name__}")
    out: list[str] = []
    for i, item in enumerate(value):
        out.append(_check_view_name(item, f"{path}[{i}]"))
    if len(out) != len(set(out)):
        raise _err(path, "duplicate entries are not allowed")
    return tuple(out)


def _check_fixed_views(value: Any, path: str, operands: set[int]) -> dict[int, str]:
    if not isinstance(value, dict):
        raise _err(path, "fixed_views must be a dict {var: view_name}")
    out: dict[int, str] = {}
    for key, view in value.items():
        if isinstance(key, bool):
            raise _err(path, "boolean variable id is not allowed")
        if isinstance(key, str) and key.lstrip("-").isdigit():
            v = int(key)
        elif isinstance(key, int):
            v = key
        else:
            raise _err(path, f"bad variable id key: {key!r}")
        check_int(v, f"{path}(var)", minimum=0)
        _check_view_name(view, f"{path}[{v}]")
        if v not in operands:
            raise _err(f"{path}[{v}]", "pinned variable must be used or defined here")
        out[v] = view
    return out


@dataclass(frozen=True)
class Instr:
    """One three-address instruction.

    ``fixed`` pins a use/def variable to a concrete physical register at THIS
    instruction point only.  The allocator materialises the pin with short
    copies around the instruction (the variable's home register may differ);
    two operands pinned to the same register at one instruction is a proven
    infeasibility.  ``clobbers`` lists registers that a call-like instruction
    destroys.

    V2 extensions (only legal in v2 programs, i.e. when ``Program`` carries
    ``var_classes``):

    * ``early_clobber``: subset of ``defs`` that are written BEFORE the
      instruction's uses are read, so an early-clobber def may not share a
      register view with any use of this instruction;
    * ``may_throw``: the instruction may transfer control to one of the
      block's ``exc_succs`` instead of completing normally; on the
      exceptional path its defs do NOT materialise (its call clobbers do);
    * ``is_call``: a call; clobbers every machine view that is not
      callee-saved (plus ``clobber_views``);
    * ``fixed_views``: like ``fixed`` but pins to a named register view;
    * ``clobber_views``: extra named views destroyed by this instruction.
    """

    uid: int
    uses: tuple[int, ...] = ()
    defs: tuple[int, ...] = ()
    fixed: dict[int, int] = field(default_factory=dict)
    clobbers: tuple[int, ...] = ()
    note: str = ""
    early_clobber: tuple[int, ...] = ()
    may_throw: bool = False
    is_call: bool = False
    fixed_views: dict[int, str] = field(default_factory=dict)
    clobber_views: tuple[str, ...] = ()

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
        # ---- v2 fields -------------------------------------------------
        early = _check_id_tuple(self.early_clobber, "early_clobber")
        bad_early = sorted(set(early) - set(defs))
        if bad_early:
            raise _err(
                "early_clobber",
                f"early-clobber variables must be defs here: {bad_early}",
            )
        object.__setattr__(self, "early_clobber", early)
        object.__setattr__(self, "may_throw", _check_bool(self.may_throw, "may_throw"))
        object.__setattr__(self, "is_call", _check_bool(self.is_call, "is_call"))
        fixed_views = _check_fixed_views(
            self.fixed_views, "fixed_views", set(uses) | set(defs)
        )
        object.__setattr__(self, "fixed_views", fixed_views)
        object.__setattr__(
            self, "clobber_views", _check_view_tuple(self.clobber_views, "clobber_views")
        )
        if fixed and fixed_views:
            raise _err("fixed", "cannot mix legacy 'fixed' and v2 'fixed_views'")
        if clobbers and self.clobber_views:
            raise _err(
                "clobbers", "cannot mix legacy 'clobbers' and v2 'clobber_views'"
            )


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
    exc_succs: tuple[int, ...] = ()

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
        exc_succs = _check_id_tuple(self.exc_succs, f"blocks[{self.label!r}].exc_succs")
        both = sorted(set(succs) & set(exc_succs))
        if both:
            raise _err(
                f"blocks[{self.label!r}].exc_succs",
                f"a successor cannot be both normal and exceptional: {both}",
            )
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
        object.__setattr__(self, "exc_succs", exc_succs)
        object.__setattr__(self, "phis", tuple(norm_phis))


@dataclass(frozen=True)
class Program:
    blocks: tuple[Block, ...]
    params: tuple[int, ...] = ()
    name: str = ""
    var_classes: dict[int, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, (list, tuple)) or not self.blocks:
            raise ProgramFormatError("blocks: expected a non-empty list/tuple")
        # A program is a "v2 program" exactly when var_classes is given.
        v2 = self.var_classes is not None
        block_limit = MAX_BLOCKS_V2 if v2 else MAX_BLOCKS
        if len(self.blocks) > block_limit:
            raise ProgramFormatError(
                f"blocks: program has {len(self.blocks)} blocks, limit {block_limit}"
            )
        blocks = tuple(self.blocks)
        for i, b in enumerate(blocks):
            if not isinstance(b, Block):
                raise _err(f"blocks[{i}]", "expected Block")
        params = _check_id_tuple(self.params or (), "params")
        if not isinstance(self.name, str):
            raise _err("name", "name must be a string")

        # ---- var_classes (v2 switch) ------------------------------------
        norm_classes: dict[int, str] | None = None
        if v2:
            if not isinstance(self.var_classes, dict):
                raise _err("var_classes", "expected a dict {var: class_name} or None")
            norm_classes = {}
            for key, cls in self.var_classes.items():
                if isinstance(key, bool):
                    raise _err("var_classes", "boolean variable id is not allowed")
                if isinstance(key, str) and key.lstrip("-").isdigit():
                    kv = int(key)
                elif isinstance(key, int):
                    kv = key
                else:
                    raise _err("var_classes", f"bad variable id key: {key!r}")
                check_int(kv, "var_classes(var)", minimum=0)
                _check_view_name(cls, f"var_classes[{kv}]")
                norm_classes[kv] = cls

        # ---- v2-only feature gating --------------------------------------
        for bi, b in enumerate(blocks):
            for ii, ins in enumerate(b.instrs):
                ip = f"blocks[{bi}].instrs[{ii}]"
                uses_v2 = (
                    ins.early_clobber
                    or ins.may_throw
                    or ins.is_call
                    or ins.fixed_views
                    or ins.clobber_views
                )
                if uses_v2 and not v2:
                    raise _err(
                        ip,
                        "v2 instruction features (early_clobber/may_throw/is_call/"
                        "fixed_views/clobber_views) require Program(var_classes=...) "
                        "and allocate2()",
                    )
                if v2 and ins.fixed:
                    raise _err(
                        f"{ip}.fixed",
                        "legacy integer 'fixed' is not allowed in v2 programs; "
                        "use fixed_views",
                    )
                if v2 and ins.clobbers:
                    raise _err(
                        f"{ip}.clobbers",
                        "legacy integer 'clobbers' is not allowed in v2 programs; "
                        "use is_call/clobber_views",
                    )
            if b.exc_succs and not v2:
                raise _err(
                    f"blocks[{bi}].exc_succs",
                    "exceptional successors require Program(var_classes=...) "
                    "and allocate2()",
                )

        n = len(blocks)
        for i, b in enumerate(blocks):
            for s in b.succs:
                if s >= n:
                    raise _err(f"blocks[{i}].succs", f"successor {s} out of range (0..{n - 1})")
            for s in b.exc_succs:
                if s >= n:
                    raise _err(
                        f"blocks[{i}].exc_succs",
                        f"exceptional successor {s} out of range (0..{n - 1})",
                    )
            # exceptional control flow contract: exactly one may_throw
            # instruction per block with exceptional successors, and no
            # may_throw instruction without exceptional successors
            throwers = [ii for ii, ins in enumerate(b.instrs) if ins.may_throw]
            if b.exc_succs and len(throwers) != 1:
                raise _err(
                    f"blocks[{i}]",
                    "a block with exc_succs must contain exactly one "
                    "may_throw instruction",
                )
            if throwers and not b.exc_succs:
                raise _err(
                    f"blocks[{i}].instrs[{throwers[0]}]",
                    "may_throw instruction requires the block to declare exc_succs",
                )

        preds: list[set[int]] = [set() for _ in range(n)]
        for i, b in enumerate(blocks):
            for s in b.succs:
                preds[s].add(i)
            for s in b.exc_succs:
                preds[s].add(i)

        # reachability from entry block 0 (normal + exceptional edges)
        seen = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            for s in blocks[cur].succs + blocks[cur].exc_succs:
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
        if norm_classes is not None:
            bad_keys = sorted(set(norm_classes) - defined)
            if bad_keys:
                raise _err(
                    "var_classes",
                    f"class given for variable(s) not in the program: {bad_keys}",
                )
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
        object.__setattr__(self, "var_classes", norm_classes)

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


# ---- dict (e.g. JSON) construction ----------------------------------------

def _at(prefix: str, i: int) -> str:
    return f"{prefix}[{i}]"


def program_from_dict(data: Any) -> Program:
    """Build a :class:`Program` from plain dicts/lists.

    Expected shape::

        {"name": str?, "params": [int]?, "var_classes": {"0": "w"}?,
         "blocks": [{"label": str?,
                     "instrs": [{"uid": int, "uses": [int], "defs": [int],
                                 "fixed": {"0": 1}, "clobbers": [int],
                                 "early_clobber": [int], "may_throw": bool,
                                 "is_call": bool, "fixed_views": {"0": "w0"},
                                 "clobber_views": ["w1"],
                                 "note": str?}],
                     "succs": [int], "exc_succs": [int]?,
                     "phis": [{"def": int, "sources": [[pred, src], ...]}]}]}

    The v2 keys are optional; a dict without them builds exactly the v1
    program it always did.
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
                        early_clobber=tuple(idata.get("early_clobber", ())),
                        may_throw=idata.get("may_throw", False),
                        is_call=idata.get("is_call", False),
                        fixed_views=dict(idata.get("fixed_views", {})),
                        clobber_views=tuple(idata.get("clobber_views", ())),
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
                exc_succs=tuple(bd.get("exc_succs", ())),
            )
        )
    params = tuple(data.get("params", ()))
    var_classes = data.get("var_classes", None)
    if var_classes is not None:
        if not isinstance(var_classes, dict):
            raise _err("var_classes", "expected an object {var: class_name}")
        var_classes = {k: v for k, v in var_classes.items()}
    try:
        return Program(
            blocks=tuple(blocks),
            params=params,
            name=str(data.get("name", "")),
            var_classes=var_classes,
        )
    except ProgramFormatError as exc:
        # make instr-level paths locatable when constructed from raw blocks
        raise exc
