"""Independent verifier used by the tests (never by the allocator itself).

It implements from scratch:

1. Its own edge-sensitive liveness for the ORIGINAL program.
2. Symbolic "reaching token" sets R(v): the set of leaf definition tokens a
   variable may hold (a phi destination is the fixpoint union of its edge
   sources -- {y, z} at a diamond join, for example).
3. A may-token abstract machine for the EMITTED program: every Reg/Slot
   holds a set of tokens; block joins are set unions, mov/load/store copy
   token sets, user instructions replace result tokens, clobbers destroy
   register contents. It checks, per original instruction occurrence:
     - each operand location holds exactly R(operand);
     - each result location ends with exactly R(def);
     - no location simultaneously carries tokens of two distinct live
       variables (would be a register collision or a leaked edge copy);
     - every independently-computed live variable is present somewhere;
     - a clobbered register never keeps a live value across the call.
4. Structural sanity of the emitted CFG (successors exist; split blocks sit
   on single edges and contain allocator glue only).
5. That the returned Liveness equals this independent computation.

Round-trip agreement is therefore an extra check; expected token sets come
from this module's own analysis, not from the allocator.
"""

from __future__ import annotations

from regalloc.model import (
    Allocation,
    MInstr,
    Program,
    Reg,
    Slot,
)


class VerificationError(AssertionError):
    pass


# --------------------------------------------------------------------------- #
# Independent edge-sensitive liveness on the original program
# --------------------------------------------------------------------------- #


def independent_liveness(program: Program):
    preds: dict[str, list[str]] = {b.id: [] for b in program.blocks}
    for b in program.blocks:
        for s in b.succs:
            preds[s].append(b.id)

    phi_dst = {b.id: {p.dst for p in b.phis} for b in program.blocks}
    edge_phi_srcs: dict[tuple[str, str], set[str]] = {}
    for b in program.blocks:
        for p in b.phis:
            for pred, src in p.srcs.items():
                edge_phi_srcs.setdefault((pred, b.id), set()).add(src)

    lin: dict[str, frozenset[str]] = {b.id: frozenset() for b in program.blocks}
    lout: dict[str, frozenset[str]] = {b.id: frozenset() for b in program.blocks}

    changed = True
    while changed:
        changed = False
        for b in reversed(program.blocks):
            cur = set()
            for s in b.succs:
                cur |= lin[s]
                cur |= edge_phi_srcs.get((b.id, s), set())
            new_out = frozenset(cur)
            for ins in reversed(b.instrs):
                cur -= set(ins.defs)
                cur |= set(ins.uses)
            new_in = frozenset(cur - phi_dst[b.id])
            # No forced union of parameters: a parameter is live only when a
            # use reaches the entry boundary (dead params stay absent).
            if new_in != lin[b.id] or new_out != lout[b.id]:
                lin[b.id] = new_in
                lout[b.id] = new_out
                changed = True

    points: dict[tuple[str, int], tuple[frozenset[str], frozenset[str]]] = {}
    for b in program.blocks:
        # Reverse walk from live-out: phi destinations that are never used
        # stay dead and never enter "before" of the first instruction.
        tmp = set(lout[b.id])
        afters: list[frozenset[str]] = []
        for ins in reversed(b.instrs):
            afters.append(frozenset(tmp))
            tmp -= set(ins.defs)
            tmp |= set(ins.uses)
        afters.reverse()
        cur = set(tmp)  # live before first instr (== after phi defs, precisely)
        for i, ins in enumerate(b.instrs):
            before = frozenset(cur)
            after = afters[i]
            points[(b.id, i)] = (before, after)
            cur = set(after)  # live after instr i == live before instr i+1

    edges = {}
    for b in program.blocks:
        for s in b.succs:
            edges[(b.id, s)] = lin[s] | edge_phi_srcs.get((b.id, s), set())
    return lin, lout, points, edges, preds


def reaching_tokens(program: Program) -> dict[str, frozenset[str]]:
    """R(v): fixpoint token set per variable (phis union edge sources)."""
    phi_list = [p for b in program.blocks for p in b.phis]
    R: dict[str, frozenset[str]] = {}
    for pname in program.params:
        R[pname] = frozenset({pname})
    for b in program.blocks:
        for ins in b.instrs:
            for d in ins.defs:
                R[d] = frozenset({d})
    for p in phi_list:
        R[p.dst] = frozenset()

    changed = True
    while changed:
        changed = False
        for phi in phi_list:
            tokens: set[str] = set()
            for src in phi.srcs.values():
                tokens |= R.get(src, frozenset({src}))
            new = frozenset(tokens)
            if new != R[phi.dst]:
                R[phi.dst] = new
                changed = True
    return R


# --------------------------------------------------------------------------- #
# Emitted-program abstract machine
# --------------------------------------------------------------------------- #


def _glue_transfer(mi: MInstr, env: dict, param_names, param_idx_box: list[int], R) -> None:
    """Transfer for allocator glue only (mov/load/store/param)."""
    if mi.opcode == "mov":
        env[mi.writes[0]] = set(env.get(mi.reads[0], ()))
        return
    if mi.opcode == "load":
        env[mi.writes[0]] = set(env.get(mi.reads[0], ()))
        return
    if mi.opcode == "store":
        env[mi.writes[0]] = set(env.get(mi.reads[0], ()))
        return
    if mi.opcode == "param":
        name = param_names[param_idx_box[0]]
        param_idx_box[0] += 1
        env[mi.writes[0]] = set(R[name])
        return
    raise VerificationError(f"unexpected opcode {mi.opcode!r} in glue transfer")


def _set_results(mi: MInstr, env: dict, R) -> None:
    # The expected result token sets for this occurrence are attached to the
    # MInstr by the verify driver through a side table keyed by (block,index).
    expected = _RESULT_TABLE.get(id(mi))
    for loc, toks in zip(mi.writes, expected):
        env[loc] = set(toks)


_RESULT_TABLE: dict[int, tuple] = {}


def _block_walk(mb, state_in, orig_block, R, program, per_point_cb=None):
    env: dict = {k: set(v) for k, v in state_in.items()}
    param_box = [0]
    param_names = program.params if mb.id == program.entry else ()
    for mi in mb.instrs:
        if per_point_cb is not None and mi.origin is not None:
            per_point_cb("before", mi, env)
        if mi.opcode in ("mov", "load", "store", "param"):
            _glue_transfer(mi, env, param_names, param_box, R)
        else:
            for r in mi.clobbers:
                if Reg(r) not in mi.writes:
                    env.pop(Reg(r), None)
            _set_results(mi, env, R)
        if per_point_cb is not None and mi.origin is not None:
            per_point_cb("after", mi, env)
    return env


def verify_allocation(program: Program, result: Allocation) -> None:
    """Raise :class:`VerificationError` if ``result`` is not equivalent."""
    K = result.K
    emitted = result.program
    R = reaching_tokens(program)
    lin, lout, points, edge_live, _preds_orig = independent_liveness(program)
    _check_returned_liveness(program, result, lin, lout, points, edge_live)

    blocks = {mb.id: mb for mb in emitted.blocks}
    if emitted.entry not in blocks:
        raise VerificationError("emitted entry block missing")
    for mb in emitted.blocks:
        for s in mb.succs:
            if s not in blocks:
                raise VerificationError(f"{mb.id!r}: unknown successor {s!r}")
    orig_index = {b.id: b for b in program.blocks}
    emitted_preds: dict[str, list[str]] = {mb.id: [] for mb in emitted.blocks}
    for mb in emitted.blocks:
        for s in mb.succs:
            emitted_preds[s].append(mb.id)
    for mb in emitted.blocks:
        if mb.id.startswith("..split."):
            if len(mb.succs) != 1 or len(emitted_preds[mb.id]) != 1:
                raise VerificationError(f"split block {mb.id!r} is not a single edge")
            for mi in mb.instrs:
                if mi.origin is not None:
                    raise VerificationError(f"split block {mb.id!r} contains user code")

    # Build the result-token side table for emitted user instructions.
    global _RESULT_TABLE
    _RESULT_TABLE = {}
    for mb in emitted.blocks:
        ob = orig_index.get(mb.id)
        if ob is None:
            continue
        for mi in mb.instrs:
            if mi.origin is not None:
                ins = ob.instrs[mi.origin]
                _RESULT_TABLE[id(mi)] = tuple(R[d] for d in ins.defs)

    # ---- may-token fixpoint over the emitted CFG --------------------------
    state_in: dict[str, frozenset] = {mb.id: frozenset() for mb in emitted.blocks}
    state_out: dict[str, frozenset] = {}

    def freeze(env: dict):
        return frozenset((k, frozenset(v)) for k, v in env.items())

    def thaw(f):
        return {k: set(v) for k, v in f}

    changed = True
    rounds = 0
    max_rounds = 1 + 4 * (len(emitted.blocks) + 1) * (len(R) + 4)
    while changed:
        changed = False
        rounds += 1
        if rounds > max_rounds:  # pragma: no cover
            raise VerificationError("emitted token analysis did not converge")
        for mb in emitted.blocks:
            seed: dict = {}
            for p in emitted_preds[mb.id]:
                for loc, toks in state_out.get(p, frozenset()):
                    seed.setdefault(loc, set()).update(toks)
            new_in = freeze(seed)
            if new_in != state_in[mb.id]:
                state_in[mb.id] = new_in
                changed = True
            env = _block_walk(mb, thaw(new_in), orig_index.get(mb.id), R, program)
            new_out = freeze(env)
            if new_out != state_out.get(mb.id):
                state_out[mb.id] = new_out
                changed = True

    # ---- per-occurrence precision on the fixpoint states ------------------
    for mb in emitted.blocks:
        ob = orig_index.get(mb.id)
        if ob is None:
            continue

        def cb(phase: str, mi: MInstr, env: dict, _mb=mb, _ob=ob) -> None:
            ins = _ob.instrs[mi.origin]
            if phase == "before":
                for v, loc in zip(ins.uses, mi.reads):
                    got = env.get(loc)
                    if got != set(R[v]):
                        raise VerificationError(
                            f"{_mb.id} origin {mi.origin} operand {v!r}: "
                            f"expected {sorted(R[v])} at {loc}, got "
                            f"{sorted(got) if got is not None else None}"
                        )
                    if v in ins.fixed_use and getattr(loc, "i", None) != ins.fixed_use[v]:
                        raise VerificationError(
                            f"{_mb.id} origin {mi.origin}: operand {v!r} violates "
                            f"fixed register r{ins.fixed_use[v]}"
                        )
                if tuple(sorted(mi.clobbers)) != tuple(sorted(ins.clobbers)):
                    raise VerificationError(
                        f"{_mb.id} origin {mi.origin}: emitted clobbers "
                        f"{mi.clobbers} != original {ins.clobbers}"
                    )
                _check_locations(
                    env, points[(_mb.id, mi.origin)][0], R,
                    f"{_mb.id} origin {mi.origin} live-before",
                )
                # Call-clobber safety is checked at "after": _block_walk
                # destroys every clobbered register (unless a result is
                # delivered there), and the live-after placement check then
                # proves every surviving value still has a valid home.
            else:
                for v, loc in zip(ins.defs, mi.writes):
                    got = env.get(loc)
                    if got != set(R[v]):
                        raise VerificationError(
                            f"{_mb.id} origin {mi.origin} result {v!r}: "
                            f"expected {sorted(R[v])} at {loc}, got {sorted(got)}"
                        )
                    if v in ins.fixed_def and getattr(loc, "i", None) != ins.fixed_def[v]:
                        raise VerificationError(
                            f"{_mb.id} origin {mi.origin}: result {v!r} violates "
                            f"fixed register r{ins.fixed_def[v]}"
                        )
                _check_locations(
                    env, points[(_mb.id, mi.origin)][1], R,
                    f"{_mb.id} origin {mi.origin} live-after",
                )

        _block_walk(mb, thaw(state_in[mb.id]), ob, R, program, cb)

    # ---- block-entry and per-edge placement on fixpoint states ------------
    split_of: dict[tuple[str, str], str] = {}
    for mb in emitted.blocks:
        if mb.id.startswith("..split."):
            (p, q) = mb.id[len("..split.") :].split("->", 1)
            split_of[(p, q)] = mb.id
    for b in program.blocks:
        mb = blocks[b.id]
        if b.id == program.entry:
            # Parameters are materialized by the leading glue (param, and for
            # spilled params a following store); check placement after the
            # whole prefix that precedes the first user instruction.
            env0 = thaw(state_in[mb.id])
            box = [0]
            for mi in mb.instrs:
                if mi.origin is not None:
                    break
                _glue_transfer(mi, env0, program.params, box, R)
            _check_locations(env0, lin[b.id], R, f"{b.id!r} after params")
        else:
            _check_locations(thaw(state_in[mb.id]), lin[b.id], R, f"{b.id!r} entry")
        for q in b.succs:
            wanted = edge_live[(b.id, q)]
            tail_id = split_of.get((b.id, q), b.id)
            _check_locations(thaw(state_out[tail_id]), wanted, R, f"edge {b.id!r}->{q!r}")

    # ---- direct register-collision assertion from independent live sets ---
    for (bid, i), (before, after) in points.items():
        for live in (before, after):
            seen: dict[int, str] = {}
            for v in live:
                reg = result.assignment.get(v)
                if reg is None:
                    continue
                if reg in seen:
                    raise VerificationError(
                        f"{bid} instr {i}: r{reg} holds both {seen[reg]!r} and {v!r}"
                    )
                seen[reg] = v

    _check_assignment_tables(program, result)


def _check_locations(env: dict, live, R, where: str) -> None:
    """Every live variable must own a distinct location.

    A variable ``v`` may live at a location whose token set contains R(v)
    (joins make phi token sets unions that include still-live leaf tokens, so
    plain set partitioning would over-report). We require a perfect bipartite
    matching from the live variables to locations containing their tokens;
    distinct live variables therefore cannot be forced onto one location.
    """
    live = sorted(live)
    locs = [loc for loc, toks in env.items() if toks]

    def homes(v):
        wanted = set(R[v])
        return [loc for loc in locs if wanted <= set(env[loc])]

    # Kuhn augmenting matching (data is tiny: <= a few dozen values).
    owner: dict = {}

    def augment(v, seen) -> bool:
        for loc in homes(v):
            if loc in seen:
                continue
            seen.add(loc)
            if loc not in owner or augment(owner[loc], seen):
                owner[loc] = v
                return True
        return False

    for v in live:
        if not augment(v, set()):
            raise VerificationError(
                f"{where}: live variable {v!r} has no collision-free location "
                f"(homes={[str(l) for l in homes(v)]})"
            )
    # Sanity: a location cannot serve two matched variables by construction
    # (matching is injective); report the concrete collision if matching
    # ambiguities left two equal-token variables sharing in a naive view.
    values = [(loc, sorted(toks)) for loc, toks in env.items() if toks]
    assert len(set(owner)) == len(owner)
    return values


def _check_assignment_tables(program: Program, result: Allocation) -> None:
    K = result.K
    originals = set(program.params)
    for b in program.blocks:
        originals.update(p.dst for p in b.phis)
        for ins in b.instrs:
            originals.update(ins.defs)
            originals.update(ins.uses)
        for p in b.phis:
            originals.update(p.srcs.values())
    if set(result.assignment) != originals:
        raise VerificationError("assignment table does not cover exactly the original variables")
    for v, r in result.assignment.items():
        if r is not None and not (isinstance(r, int) and 0 <= r < K):
            raise VerificationError(f"assignment[{v!r}] is not a register in [0,{K - 1}]")
    slots = list(result.stack_slots.values())
    if sorted(slots) != list(range(len(slots))):
        raise VerificationError("stack slots must be contiguous 0..n-1")
    # A variable that is provably never live (dead code / unused parameter)
    # needs neither a register nor a slot. Everything else mapped to None
    # must have a stack slot.
    lin, lout, points, edges, _ = independent_liveness(program)
    ever_live: set[str] = set()
    for s in lin.values():
        ever_live |= set(s)
    for s in lout.values():
        ever_live |= set(s)
    for before, after in points.values():
        ever_live |= set(before) | set(after)
    for s in edges.values():
        ever_live |= set(s)
    for v, i in result.stack_slots.items():
        if result.assignment[v] is not None:
            raise VerificationError(f"spilled variable {v!r} also assigned a register")
    for v, r in result.assignment.items():
        if r is None and v not in result.stack_slots and v in ever_live:
            raise VerificationError(
                f"live variable {v!r} has neither register nor stack slot"
            )


def _check_returned_liveness(program, result: Allocation, lin, lout, points, edges) -> None:
    live = result.live

    def tup(s):
        return tuple(sorted(s))

    for b in program.blocks:
        if live.live_in[b.id] != tup(lin[b.id]):
            raise VerificationError(f"live_in mismatch at {b.id!r}")
        if live.live_out[b.id] != tup(lout[b.id]):
            raise VerificationError(f"live_out mismatch at {b.id!r}")
        for i in range(len(b.instrs)):
            exp = (tup(points[(b.id, i)][0]), tup(points[(b.id, i)][1]))
            if live.points[(b.id, i)] != exp:
                raise VerificationError(f"liveness point mismatch at {(b.id, i)}")
    for edge, toks in edges.items():
        if live.edge_live[edge] != tup(toks):
            raise VerificationError(f"edge liveness mismatch at {edge}")
