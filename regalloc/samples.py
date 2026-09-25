"""Deterministic sample programs and a seeded program generator.

Nothing here is random at run time: hand-built fixtures are constants and the
generator is driven by :class:`random.Random` with a fixed seed, so every
run reproduces the exact same program.
"""

from __future__ import annotations

import random

from .model import Block, Instr, PhiNode, Program


def sample_call_surviving() -> Program:
    """A value is live across a call that clobbers r0 and r1.

    Layout::

        v1 = f()
        v2 = call(v1, v0)        # clobbers r0,r1
        v3 = f(v1, v2)           # v1 had to survive the call
    """
    b0 = Block(
        label="entry",
        instrs=(
            Instr(uid=0, uses=(), defs=(1,), note="v1 = const"),
            Instr(uid=1, uses=(1, 0), defs=(2,), clobbers=(0, 1), note="call f"),
            Instr(uid=2, uses=(1, 2), defs=(3,), note="v3 = add(v1,v2)"),
        ),
        succs=(),
    )
    return Program(blocks=(b0,), params=(0,), name="call-surviving")


def sample_critical_phi() -> Program:
    """Merge phi reached over both critical and ordinary edges.

    ::

        b0 ---> b1 --+--> b3(phi)
         \\           /
          +--> b3   b2 --> b3

    Edges b0->b3 and b1->b3 are critical (predecessor has >1 successor,
    b3 has >1 predecessors) and must be split; b2->b3 is not critical.
    """
    b0 = Block(
        label="b0",
        instrs=(Instr(uid=0, uses=(0,), defs=(1,), note="v1 = f(v0)"),),
        succs=(1, 3),
    )
    b1 = Block(
        label="b1",
        instrs=(Instr(uid=1, uses=(1,), defs=(2,), note="v2 = g(v1)"),),
        succs=(2, 3),
    )
    b2 = Block(
        label="b2",
        instrs=(Instr(uid=2, uses=(2,), defs=(3,), note="v3 = h(v2)"),),
        succs=(3,),
    )
    b3 = Block(
        label="merge",
        instrs=(Instr(uid=3, uses=(4,), defs=(5,), note="use phi v4"),),
        succs=(),
        phis=(PhiNode(def_var=4, sources=((0, 1), (1, 2), (2, 3))),),
    )
    return Program(blocks=(b0, b1, b2, b3), params=(0,), name="critical-phi")


def sample_fixed_collision() -> Program:
    """Two simultaneous operands are both pinned to r1 (provably impossible)."""
    b0 = Block(
        label="entry",
        instrs=(
            Instr(
                uid=0,
                uses=(0, 1),
                defs=(),
                fixed={0: 1, 1: 1},
                note="two operands both demand r1",
            ),
        ),
        succs=(),
    )
    return Program(blocks=(b0,), params=(0, 1), name="fixed-collision")


def sample_register_pressure() -> Program:
    """Five values are simultaneously live; with K=2 spilling is mandatory.

    All five values coexist at the peak (v0 + v1..v4), but every instruction
    reads at most two operands, so spilling genuinely resolves the pressure
    (unlike an impossible 5-operand instruction)::

        v1..v4 defined      -> live: v0,v1,v2,v3,v4 (peak 5)
        v5 = add(v1, v2)    -> v3,v4,v5,v0 live
        v6 = add(v3, v4)
        v7 = add(v5, v6)
        v8 = add(v7, v0)
    """
    b0 = Block(
        label="entry",
        instrs=(
            Instr(uid=0, uses=(), defs=(1,), note="def v1"),
            Instr(uid=1, uses=(), defs=(2,), note="def v2"),
            Instr(uid=2, uses=(), defs=(3,), note="def v3"),
            Instr(uid=3, uses=(), defs=(4,), note="def v4"),
            Instr(uid=4, uses=(1, 2), defs=(5,), note="pair 1"),
            Instr(uid=5, uses=(3, 4), defs=(6,), note="pair 2"),
            Instr(uid=6, uses=(5, 6), defs=(7,), note="join"),
            Instr(uid=7, uses=(7, 0), defs=(8,), note="with param"),
            Instr(uid=8, uses=(8,), defs=(), note="sink"),
        ),
        succs=(),
    )
    return Program(blocks=(b0,), params=(0,), name="register-pressure")


def sample_loop_call() -> Program:
    """Loop with a call inside: a loop-carried value survives the call.

    ::

        b0: v1 = init(v0)
        b1(phi v2 <- v1 | v5): v3 = call(v2) clobber r0; v4 = f(v3);
                               v5 = f(v2,v4); cond -> b1 | b2
        b2: v6 = ret(v2)
    """
    b0 = Block(
        label="entry",
        instrs=(Instr(uid=0, uses=(0,), defs=(1,), note="init"),),
        succs=(1,),
    )
    b1 = Block(
        label="loop",
        instrs=(
            Instr(uid=1, uses=(2,), defs=(3,), clobbers=(0,), note="call in loop"),
            Instr(uid=2, uses=(3,), defs=(4,), note="step1"),
            Instr(uid=3, uses=(2, 4), defs=(5,), note="step2"),
        ),
        succs=(1, 2),
        phis=(PhiNode(def_var=2, sources=((0, 1), (1, 5))),),
    )
    b2 = Block(
        label="exit",
        instrs=(Instr(uid=4, uses=(2,), defs=(6,), note="result"),),
        succs=(),
    )
    return Program(blocks=(b0, b1, b2), params=(0,), name="loop-call")


def sample_hot_loop_cold_error() -> Program:
    """Hot loop with a cold error-recovery arm inside the same loop nest.

    Statically the cold arm sits at the same loop depth as the hot latch, so
    the legacy loop-depth heuristic cannot tell them apart; only an explicit
    block-frequency profile separates them.  Two seeds are live across the
    whole loop: ``v2`` (cold recovery seed, used three times but only on the
    error arm) and ``v15`` (hot seed, used three times in the latch).

    Parameters: ``v0`` step, ``v1`` termination limit, ``v25`` error
    threshold.  Blocks::

        0 entry  : v2=7, v3=0, v15=1                   -> 1
        1 loop   : phi v4 = v3 | v8
                   v5 = v4 + v0(step)
                   v7 = (v25(threshold) < v5)          # overflow => error
                   br2 v7: !=0 -> 3(cold) else 2(latch)
        3 cold   : v9  = v5 + v0
                   v10 = v9 + v2
                   v11 = v10 + v2
                   v12 = v11 + v2                      -> 2
        2 latch  : phi v6 = v5 | v12
                   v8  = v6 + v0
                   v16 = v8 + v15
                   v17 = v16 + v15
                   v18 = v17 + v15
                   v13 = (v8 < v1)
                   br2 v13: !=0 -> 1(loop) else 4(exit)
        4 exit   : v14 = v8 + v1

    Concrete runs (see regalloc.concrete): ``(step, limit, threshold)`` =
    ``(3, 100, 1000)`` stays on the hot path and terminates; ``(300, 1000,
    100)`` enters the cold recovery arm (twice) before terminating.  Both
    paths are exercised by the demo and the tests.
    """
    b0 = Block(
        label="entry",
        instrs=(
            Instr(uid=0, uses=(), defs=(2,), note="const:7"),   # cold seed
            Instr(uid=1, uses=(), defs=(3,), note="const:0"),   # acc init
            Instr(uid=2, uses=(), defs=(15,), note="const:1"),  # hot seed
        ),
        succs=(1,),
    )
    b1 = Block(
        label="loop",
        instrs=(
            Instr(uid=3, uses=(4, 0), defs=(5,), note="add"),
            Instr(uid=4, uses=(25, 5), defs=(7,), note="lt"),
            Instr(uid=5, uses=(7,), defs=(), note="br2"),
        ),
        succs=(3, 2),
        phis=(PhiNode(def_var=4, sources=((0, 3), (2, 8))),),
    )
    b3 = Block(
        label="cold",
        instrs=(
            Instr(uid=6, uses=(5, 0), defs=(9,), note="add"),
            Instr(uid=7, uses=(9, 2), defs=(10,), note="add"),
            Instr(uid=8, uses=(10, 2), defs=(11,), note="add"),
            Instr(uid=9, uses=(11, 2), defs=(12,), note="add"),
        ),
        succs=(2,),
    )
    b2 = Block(
        label="latch",
        instrs=(
            Instr(uid=10, uses=(6, 0), defs=(8,), note="add"),
            Instr(uid=11, uses=(8, 15), defs=(16,), note="add"),
            Instr(uid=12, uses=(16, 15), defs=(17,), note="add"),
            Instr(uid=13, uses=(17, 15), defs=(18,), note="add"),
            Instr(uid=14, uses=(8, 1), defs=(13,), note="lt"),
            Instr(uid=15, uses=(13,), defs=(), note="br2"),
        ),
        succs=(1, 4),
        phis=(PhiNode(def_var=6, sources=((1, 5), (3, 12))),),
    )
    b4 = Block(
        label="exit",
        instrs=(Instr(uid=16, uses=(8, 1), defs=(14,), note="add"),),
        succs=(),
    )
    return Program(
        blocks=(b0, b1, b2, b3, b4),
        params=(0, 1, 25),
        name="hot-loop-cold-error",
    )


# Frequency profiles for the fixture above, one entry per block index
# (entry, loop, latch, cold, exit).  HOT_PROFILE models a loop body executed
# ~1000x with a rare error arm; COLD_PROFILE is the exact inversion (the
# recovery arm is the hot path and the latch is cold).
HOT_PROFILE = (1, 1000, 1000, 1, 1)
COLD_PROFILE = (1, 1, 1, 1000, 1)
HOT_PARAMS = {0: 3, 1: 100, 25: 1000}
COLD_PARAMS = {0: 300, 1: 1000, 25: 100}


def generate_pressure_program(seed: int = 20260922, *, width: int = 5,
                              clobber: bool = True) -> Program:
    """Deterministically generated straight-line pressure program.

    ``width`` independent values v1..vwidth are defined up front (they all
    coexist with parameter v0), then consumed pairwise in a fixed order.  At
    most two operands are read per instruction, so spilling can resolve the
    pressure.  When ``clobber`` is set, a call clobbering r0 is inserted
    while several values are still live.  The PRNG is seeded, so the output
    is fully reproducible.
    """
    rng = random.Random(seed)
    uid = 0
    instrs: list[Instr] = []
    for i in range(width):
        v = i + 1
        instrs.append(Instr(uid=uid, uses=(), defs=(v,), note=f"gen v{v}"))
        uid += 1

    next_var = width + 1
    alive = list(range(1, width + 1))
    if clobber:
        # call consumes the first two live values, returns a fresh value,
        # and clobbers r0 while the remaining values must survive
        used = alive[:2]
        alive = alive[2:]
        call_res = next_var
        next_var += 1
        instrs.append(
            Instr(uid=10_000, uses=tuple(used), defs=(call_res,),
                  clobbers=(0,), note="generated call")
        )
        alive.append(call_res)
        rng.shuffle(alive)

    # pairwise reduce until one value remains (deterministic shuffled order)
    rng.shuffle(alive)
    while len(alive) > 1:
        a, b = alive.pop(), alive.pop()
        instrs.append(Instr(uid=uid, uses=(a, b), defs=(next_var,), note="pair"))
        uid += 1
        alive.append(next_var)
        next_var += 1
    final = alive[0]
    instrs.append(Instr(uid=uid, uses=(final, 0), defs=(next_var,), note="with param"))
    uid += 1
    instrs.append(Instr(uid=uid, uses=(next_var,), defs=(), note="sink"))
    b0 = Block(label="gen", instrs=tuple(instrs), succs=())
    return Program(blocks=(b0,), params=(0,), name=f"generated-pressure-{seed}")
