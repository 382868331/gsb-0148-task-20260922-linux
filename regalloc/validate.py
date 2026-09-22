"""Strict, locatable validation of the public input contract.

Nothing here performs allocation; every check raises either
:class:`InvalidArgumentError` (wrong scalar type for an API argument) or
:class:`InvalidProgramError` (malformed IR, with a ``location``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .model import Block, Instruction, Phi, Program, InvalidArgumentError, InvalidProgramError


def _is_int(x: Any) -> bool:
    # bool is a subclass of int; the contract rejects booleans where integers
    # are expected.
    return isinstance(x, int) and not isinstance(x, bool)


def check_finite_number(x: Any, what: str) -> float:
    """Reject NaN/Infinity (and non-numbers); used for every numeric input."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise InvalidArgumentError(f"{what} must be a real number, got {type(x).__name__}")
    f = float(x)
    if not math.isfinite(f):
        raise InvalidArgumentError(f"{what} must be finite, got {x!r}")
    return f


def _is_reg(x: Any, K: int) -> bool:
    return _is_int(x) and 0 <= x < K


def _ident(x: Any) -> bool:
    return isinstance(x, str) and x != ""


def _var_ident(x: Any) -> bool:
    # ".." is reserved for allocator-synthetic temporaries / split blocks.
    return _ident(x) and not x.startswith("..")


def _check_reg(x: Any, K: int, location: str, field_name: str) -> None:
    if not _is_reg(x, K):
        raise InvalidProgramError(
            f"{field_name} must be an int register in [0,{K - 1}] (bools rejected), got {x!r}",
            location,
        )


def _check_tuple_of_idents(value: Any, K: int, location: str, field_name: str, *, unique: bool) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise InvalidProgramError(f"{field_name} must be a tuple, got {type(value).__name__}", location)
    out: list[str] = []
    for j, v in enumerate(value):
        if not _var_ident(v):
            raise InvalidProgramError(
                f"{field_name}[{j}] must be a non-empty str not starting with '..' (reserved), got {v!r}",
                location,
            )
        out.append(v)
    if unique and len(set(out)) != len(out):
        raise InvalidProgramError(f"{field_name} contains duplicates: {out}", location)
    return tuple(out)


def validate_program(program: Any, K: Any) -> Program:
    """Validate ``program`` and register count ``K``; return the Program as-is."""
    if not _is_int(K):
        raise InvalidArgumentError(f"K must be an int (bools rejected), got {type(K).__name__}")
    if not 2 <= K <= 8:
        raise InvalidArgumentError(f"K must be in 2..8, got {K}")

    if not isinstance(program, Program):
        raise InvalidArgumentError(f"program must be a Program instance, got {type(program).__name__}")

    blocks = program.blocks
    if not isinstance(blocks, tuple) or not blocks:
        raise InvalidProgramError("blocks must be a non-empty tuple", "program")
    if len(blocks) > 40:
        raise InvalidProgramError(f"at most 40 blocks allowed, got {len(blocks)}", "program")

    ids: set[str] = set()
    block_map: dict[str, Block] = {}
    for b in blocks:
        if not isinstance(b, Block):
            raise InvalidProgramError(f"blocks element must be a Block, got {type(b).__name__}", "program")
        if not _var_ident(b.id):
            raise InvalidProgramError(
                "block id must be a non-empty str not starting with '..' (reserved)",
                "program",
            )
        if b.id in ids:
            raise InvalidProgramError(f"duplicate block id {b.id!r}", "program")
        ids.add(b.id)
        block_map[b.id] = b

    if not _ident(program.entry) or program.entry not in ids:
        raise InvalidProgramError(f"entry {program.entry!r} is not a block id", "program")

    params = _check_tuple_of_idents(program.params, K, "program", "params", unique=True)

    # Structural per-block validation.
    defined: set[str] = set()
    used: set[str] = set()

    def _demand_var(v: str, location: str) -> None:
        used.add(v)

    for b in blocks:
        loc = f"block {b.id!r}"

        if not isinstance(b.succs, tuple):
            raise InvalidProgramError("succs must be a tuple", loc)
        for s in b.succs:
            if not _ident(s) or s not in ids:
                raise InvalidProgramError(f"successor {s!r} is not a known block id", loc)
        if len(set(b.succs)) != len(b.succs):
            raise InvalidProgramError(f"duplicate successors: {b.succs}", loc)

        phi_dsts: set[str] = set()
        if not isinstance(b.phis, tuple):
            raise InvalidProgramError("phis must be a tuple", loc)
        for pi, phi in enumerate(b.phis):
            ploc = f"{loc} phi[{pi}]"
            if not isinstance(phi, Phi):
                raise InvalidProgramError(f"expected Phi, got {type(phi).__name__}", ploc)
            if not _var_ident(phi.dst):
                raise InvalidProgramError("phi dst must be a non-empty str not starting with '..'", ploc)
            if phi.dst in phi_dsts:
                raise InvalidProgramError(f"duplicate phi destination {phi.dst!r}", ploc)
            phi_dsts.add(phi.dst)
            if phi.fixed_def is not None:
                _check_reg(phi.fixed_def, K, ploc, "fixed_def")
            if not isinstance(phi.srcs, Mapping):
                raise InvalidProgramError("phi srcs must be a mapping pred_id -> var", ploc)
            for pk, pv in phi.srcs.items():
                if not _ident(pk) or pk.startswith(".."):
                    raise InvalidProgramError("phi src key (predecessor id) must not start with '..'", ploc)
                if not _var_ident(pv):
                    raise InvalidProgramError(f"phi src for pred {pk!r} must be a non-empty str not starting with '..'", ploc)
                _demand_var(pv, ploc)

        if not isinstance(b.instrs, tuple):
            raise InvalidProgramError("instrs must be a tuple", loc)
        for ii, ins in enumerate(b.instrs):
            iloc = f"{loc} instr[{ii}]"
            if not isinstance(ins, Instruction):
                raise InvalidProgramError(f"expected Instruction, got {type(ins).__name__}", iloc)
            if not _ident(ins.opcode):
                raise InvalidProgramError("opcode must be a non-empty str", iloc)
            if ins.opcode in ("mov", "load", "store", "param"):
                raise InvalidProgramError(
                    f"opcode {ins.opcode!r} is reserved for allocator-emitted glue",
                    iloc,
                )
            uses = _check_tuple_of_idents(ins.uses, K, iloc, "uses", unique=True)
            defs = _check_tuple_of_idents(ins.defs, K, iloc, "defs", unique=True)
            for u in uses:
                _demand_var(u, iloc)

            if not isinstance(ins.fixed_use, Mapping):
                raise InvalidProgramError("fixed_use must be a mapping var -> register", iloc)
            for fv, fr in ins.fixed_use.items():
                if fv not in uses:
                    raise InvalidProgramError(f"fixed_use variable {fv!r} is not in uses {uses}", iloc)
                _check_reg(fr, K, iloc, f"fixed_use[{fv!r}]")
            if not isinstance(ins.fixed_def, Mapping):
                raise InvalidProgramError("fixed_def must be a mapping var -> register", iloc)
            for fv, fr in ins.fixed_def.items():
                if fv not in defs:
                    raise InvalidProgramError(f"fixed_def variable {fv!r} is not in defs {defs}", iloc)
                _check_reg(fr, K, iloc, f"fixed_def[{fv!r}]")
            if not isinstance(ins.clobbers, tuple):
                raise InvalidProgramError("clobbers must be a tuple of registers", iloc)
            seen_cl: set[int] = set()
            for cr in ins.clobbers:
                _check_reg(cr, K, iloc, "clobbers element")
                if cr in seen_cl:
                    raise InvalidProgramError(f"duplicate clobbered register r{cr}", iloc)
                seen_cl.add(cr)
            for d in defs:
                if d in defined:
                    raise InvalidProgramError(
                        f"variable {d!r} is defined more than once",
                        iloc,
                    )
                defined.add(d)

    # phi destinations are definitions too; check global uniqueness.
    for b in blocks:
        for pi, phi in enumerate(b.phis):
            if phi.dst in defined:
                raise InvalidProgramError(
                    f"variable {phi.dst!r} is defined more than once",
                    f"block {b.id!r} phi[{pi}]",
                )
            defined.add(phi.dst)

    for p in params:
        if p in defined:
            raise InvalidProgramError(f"parameter {p!r} must not also be defined by an instruction/phi", "program")

    # Every used or phi-sourced variable must exist (param or a definition).
    known = defined | set(params)
    unknown = sorted(used - known)
    if unknown:
        raise InvalidProgramError(
            f"used variables that are neither parameters nor defined anywhere: {unknown}",
            "program",
        )

    # Phi src keys must equal the predecessor set exactly, once predecessors
    # are known.
    preds: dict[str, list[str]] = {b.id: [] for b in blocks}
    for b in blocks:
        for s in b.succs:
            preds[s].append(b.id)
    entry_preds = preds[program.entry]
    if entry_preds:
        raise InvalidProgramError(
            f"entry block {program.entry!r} must have no CFG predecessors, got {sorted(entry_preds)}",
            "program",
        )

    # Reachability from entry (locatable rejection of orphan blocks).
    seen: set[str] = set()
    stack = [program.entry]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(block_map[cur].succs)
    unreachable = [b.id for b in blocks if b.id not in seen]
    if unreachable:
        raise InvalidProgramError(
            f"blocks not reachable from the entry block: {unreachable}",
            "program",
        )

    # Dominators (iterative fixedpoint), used for def-before-use checks.
    dom: dict[str, set[str]] = {b.id: set(ids) for b in blocks}
    dom[program.entry] = {program.entry}
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if b.id == program.entry:
                continue
            ps = preds[b.id]
            new = set.intersection(*(dom[p] for p in ps)) if ps else set()
            new.add(b.id)
            if new != dom[b.id]:
                dom[b.id] = new
                changed = True

    def dominates(d_block: str, u_block: str) -> bool:
        return d_block in dom[u_block]

    # Locate the unique definition of every variable.
    DefInfo = tuple  # (block, kind, index)  kind: "phi" | "instr" | "param"
    def_site: dict[str, tuple[str, str, int]] = {}
    for p in params:
        def_site[p] = (program.entry, "param", -1)
    for b in blocks:
        for pi, phi in enumerate(b.phis):
            def_site[phi.dst] = (b.id, "phi", pi)
        for ii, ins in enumerate(b.instrs):
            for d in ins.defs:
                def_site[d] = (b.id, "instr", ii)

    def check_use(v: str, u_block: str, u_index: int, on_edge_from, location: str) -> None:
        if v not in def_site:
            return  # already reported as unknown
        d_block, d_kind, d_index = def_site[v]
        if on_edge_from is not None:
            # Value is read at the tail of the predecessor block.
            if not dominates(d_block, on_edge_from):
                raise InvalidProgramError(
                    f"variable {v!r} is used on edge {on_edge_from!r}->{u_block!r} "
                    f"but its definition in {d_block!r} does not dominate that edge",
                    location,
                )
            return
        if not dominates(d_block, u_block):
            raise InvalidProgramError(
                f"variable {v!r} used in {u_block!r} is not defined on every "
                f"path (def in {d_block!r} does not dominate the use); use a phi",
                location,
            )
        if d_block == u_block and d_kind == "instr":
            if d_index >= u_index:
                raise InvalidProgramError(
                    f"variable {v!r} is used at instr {u_index} of {u_block!r} "
                    f"but defined later at instr {d_index}",
                    location,
                )

    for b in blocks:
        for pi, phi in enumerate(b.phis):
            ploc = f"block {b.id!r} phi[{pi}]"
            if not preds[b.id]:
                raise InvalidProgramError(
                    "block with phis must have at least one predecessor", ploc
                )
            keys = set(phi.srcs.keys())
            if keys != set(preds[b.id]):
                raise InvalidProgramError(
                    f"phi srcs predecessors {sorted(keys)} != block predecessors {sorted(preds[b.id])}",
                    ploc,
                )
            for pred in phi.srcs:
                check_use(phi.srcs[pred], b.id, pi, on_edge_from=pred, location=ploc)
        for ii, ins in enumerate(b.instrs):
            iloc = f"block {b.id!r} instr[{ii}]"
            for u in ins.uses:
                check_use(u, b.id, ii, on_edge_from=None, location=iloc)

    return program
