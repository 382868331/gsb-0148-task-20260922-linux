"""Deterministic SSA-CFG generator for property tests.

A tiny fixed-seed LCG drives a structured builder that only produces
dominance-valid programs: a chain of sections, each a straight-line run, a
diamond (with a phi), or a counted loop (with two header phis). Uses always
reference names dominating the point; clobbers/fixed registers are added
randomly but never make a program structurally infeasible.
"""

from __future__ import annotations

from regalloc.model import Block, Instruction, Phi, Program


class _LCG:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF

    def u32(self) -> int:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s

    def below(self, n: int) -> int:
        return self.u32() % n


def generate_program(seed: int) -> Program:
    rng = _LCG(seed)
    counter = 0

    def fresh(prefix: str = "v") -> str:
        nonlocal counter
        counter += 1
        # The underscore makes names from different prefixes (e.g. "i" and
        # "i2") unable to coincide.
        return f"{prefix}_{counter}"

    params = tuple(fresh("p") for _ in range(rng.below(4)))

    blocks: list[Block] = []

    class IdGen:
        def __init__(self):
            self.n = -1

        def __call__(self) -> str:
            self.n += 1
            return f"b{self.n}"

    nid = IdGen()
    entry = nid()

    def straight(available, at_least=1, *, allow_call=True, force_defs=()):
        """Dominance-valid straight-line code. Returns (instrs, env)."""
        instrs: list[Instruction] = []
        env = list(available)
        forced = list(force_defs)
        for _ in range(at_least + rng.below(3)):
            d = forced.pop(0) if forced else fresh()
            uses: list[str] = []
            if env and rng.below(2):
                uses.append(env[rng.below(len(env))])
                if len(env) >= 2 and rng.below(2):
                    u = env[rng.below(len(env))]
                    if u not in uses:
                        uses.append(u)
            clobbers: tuple[int, ...] = ()
            opcode = "op"
            if allow_call and rng.below(4) == 0:
                opcode = "call"
                clobbers = tuple(r for r in (0, 1) if rng.below(2))
            instrs.append(Instruction(opcode, tuple(uses), (d,), clobbers=clobbers))
            env.append(d)
        for d in forced:
            uses = (env[0],) if env else ()
            instrs.append(Instruction("op", uses, (d,)))
            env.append(d)
        return tuple(instrs), env

    # Seed entry with a couple of values, then chain sections.
    seed_instrs, env = straight(params, at_least=2)
    blocks.append(Block(entry, instrs=seed_instrs, succs=()))
    tail = entry
    tail_live = env

    sections = 2 + rng.below(4)
    for _ in range(sections):
        kind = rng.below(3)
        join = nid()

        if kind == 0:  # straight line
            instrs, env = straight(tail_live)
            blocks.append(Block(join, instrs=instrs, succs=()))
            _link(tail, join, blocks)
            tail, tail_live = join, env

        elif kind == 1:  # diamond with a phi
            br, left, right = nid(), nid(), nid()
            cond = fresh("c")
            blocks.append(
                Block(br, instrs=(Instruction("tst", tuple(tail_live[:1]), (cond,)),),
                      succs=(left, right))
            )
            _link(tail, br, blocks)
            li, lenv = straight(tail_live)
            ri, renv = straight(tail_live)
            blocks.append(Block(left, instrs=li, succs=(join,)))
            blocks.append(Block(right, instrs=ri, succs=(join,)))
            m = fresh("m")
            blocks.append(
                Block(
                    join,
                    phis=(Phi(m, {left: lenv[-1], right: renv[-1]}),),
                    instrs=(),
                    succs=(),
                )
            )
            tail, tail_live = join, list(dict.fromkeys(tail_live + [m]))

        else:  # counted loop with two header phis
            head, body = nid(), nid()
            iv0, ac0 = fresh("z"), fresh("z")
            pre, pre_env = straight(tail_live, force_defs=(iv0, ac0))
            pre_id = nid()
            blocks.append(Block(pre_id, instrs=pre, succs=(head,)))
            _link(tail, pre_id, blocks)
            iv, ac, cond = fresh("i"), fresh("acc"), fresh("c")
            ivn, acn = fresh("i2"), fresh("a2")
            blocks.append(
                Block(
                    head,
                    phis=(
                        Phi(iv, {pre_id: iv0, body: ivn}),
                        Phi(ac, {pre_id: ac0, body: acn}),
                    ),
                    instrs=(Instruction("cmp", (iv,), (cond,)),),
                    succs=(body, join),
                )
            )
            bi, _ = straight(list(dict.fromkeys(tail_live + [iv, ac])),
                             force_defs=(ivn, acn))
            blocks.append(Block(body, instrs=bi, succs=(head,)))
            blocks.append(Block(join, instrs=(), succs=()))
            tail, tail_live = join, list(dict.fromkeys(tail_live + [ac]))

    # Return block.
    ret = nid()
    _link(tail, ret, blocks)
    uses = tail_live[: min(2, len(tail_live))]
    if not uses:
        rv = fresh("rv")
        blocks.append(Block(ret, instrs=(Instruction("li", (), (rv,)), Instruction("ret", (rv,), ())), succs=()))
    else:
        blocks.append(Block(ret, instrs=(Instruction("ret", tuple(uses), ()),), succs=()))

    return Program(entry=entry, blocks=tuple(blocks), params=params)


def _link(src: str, dst: str, blocks: list[Block]) -> None:
    for i, b in enumerate(blocks):
        if b.id == src:
            blocks[i] = Block(b.id, b.phis, b.instrs, b.succs + (dst,))
            return
    raise AssertionError(f"internal generator error: block {src!r} missing")
