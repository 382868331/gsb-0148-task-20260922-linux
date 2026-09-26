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


def sample_branch_slot_reuse() -> Program:
    """Contrast fixture for stack-slot packing.

    Two mutually exclusive branches each spill branch-private values whose
    lifetimes never overlap; after lowering they may share one physical stack
    offset.  Two values (v1, v2) are defined before the branch and stay live
    through both arms up to the join: they cannot share with anyone that
    overlaps them, and values arriving at the join (phis v11, v12) cannot
    reuse slots of values that are still live across the join.

    ::

        entry: v1, v2 -> left | right
        left:  v3,v4 private; v5=f(v3,v1); v6=f(v4,v2)
        right: v7,v8 private; v9=f(v7,v1); v10=f(v8,v2)
        join:  phi v11 <- v5|v9 ; phi v12 <- v6|v10
               v13=f(v11,v12); sink(v13,v0)
    """
    b0 = Block(
        label="entry",
        instrs=(
            Instr(uid=0, uses=(), defs=(1,), note="v1"),
            Instr(uid=1, uses=(), defs=(2,), note="v2"),
        ),
        succs=(1, 2),
    )
    b1 = Block(
        label="left",
        instrs=(
            Instr(uid=2, uses=(), defs=(3,), note="v3 private"),
            Instr(uid=3, uses=(), defs=(4,), note="v4 private"),
            Instr(uid=4, uses=(3, 1), defs=(5,), note="v5=f(v3,v1)"),
            Instr(uid=5, uses=(4, 2), defs=(6,), note="v6=f(v4,v2)"),
        ),
        succs=(3,),
    )
    b2 = Block(
        label="right",
        instrs=(
            Instr(uid=6, uses=(), defs=(7,), note="v7 private"),
            Instr(uid=7, uses=(), defs=(8,), note="v8 private"),
            Instr(uid=8, uses=(7, 1), defs=(9,), note="v9=f(v7,v1)"),
            Instr(uid=9, uses=(8, 2), defs=(10,), note="v10=f(v8,v2)"),
        ),
        succs=(3,),
    )
    b3 = Block(
        label="join",
        instrs=(
            Instr(uid=10, uses=(11, 12), defs=(13,), note="join use"),
            Instr(uid=11, uses=(13, 0), defs=(), note="sink"),
        ),
        succs=(),
        phis=(
            PhiNode(def_var=11, sources=((1, 5), (2, 9))),
            PhiNode(def_var=12, sources=((1, 6), (2, 10))),
        ),
    )
    return Program(blocks=(b0, b1, b2, b3), params=(0,), name="branch-slot-reuse")


def sample_spilled_across_call() -> Program:
    """Several values are live across a call and one must be spilled.

    The spilled value survives the call *in memory*: the frame slot must not
    be reused by anything else until the value is consumed afterwards, but a
    dead slot may legally be reused across the call itself (calls clobber
    registers, not stack slots).

    ::

        v1..v4 defined; call clobbers r0,r1 while v1..v4 survive;
        v5=f(v1,v2); v6=f(v3,v4); sink(v5,v6)
    """
    b0 = Block(
        label="entry",
        instrs=(
            Instr(uid=0, uses=(), defs=(1,), note="v1"),
            Instr(uid=1, uses=(), defs=(2,), note="v2"),
            Instr(uid=2, uses=(), defs=(3,), note="v3"),
            Instr(uid=3, uses=(), defs=(4,), note="v4"),
            Instr(uid=4, uses=(1, 2), defs=(5,), clobbers=(0, 1), note="call(v1,v2)"),
            Instr(uid=5, uses=(3, 4), defs=(6,), note="v6=f(v3,v4)"),
            Instr(uid=6, uses=(5, 6), defs=(), note="sink"),
        ),
        succs=(),
    )
    return Program(blocks=(b0,), params=(), name="spilled-across-call")


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
