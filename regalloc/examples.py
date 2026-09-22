"""Deterministic example programs used by the demo and the test-suite.

Every program is built by plain code (no randomness, no external data), so
the repository alone is enough to reproduce any result.
"""

from __future__ import annotations

from .model import Block, Instruction, Phi, Program


def critical_edge_phi_program() -> Program:
    """A diamond whose branch block jumps to two different joins.

    The edge ``branch -> join`` is critical (branch has two successors, join
    has two predecessors), so the allocator must split it before placing the
    phi copies.
    """
    return Program(
        entry="entry",
        params=("p",),
        blocks=(
            Block(
                "entry",
                instrs=(Instruction("start", ("p",), ("v0",)),),
                succs=("branch",),
            ),
            Block(
                "branch",
                instrs=(Instruction("tst", ("v0",), ("c",)),),
                succs=("left", "join"),
            ),
            Block(
                "left",
                instrs=(Instruction("inc", ("v0",), ("v1",)),),
                succs=("join",),
            ),
            Block(
                "join",
                phis=(Phi("m", {"branch": "c", "left": "v1"}),),
                instrs=(Instruction("out", ("m", "v0"), ("w",)),),
                succs=("end",),
            ),
            Block("end", instrs=(Instruction("ret", ("w",), ()),), succs=()),
        ),
    )


def call_pressure_program() -> Program:
    """Four long-lived values with a call that clobbers r0 and r1.

    K=2 forces spilling; K=3 still has to keep live-across values out of the
    clobbered registers.
    """
    return Program(
        entry="entry",
        params=("a", "b"),
        blocks=(
            Block(
                "entry",
                instrs=(
                    Instruction("li1", (), ("t1",)),
                    Instruction("li2", (), ("t2",)),
                    Instruction("call", ("a",), ("cr",), clobbers=(0, 1)),
                    Instruction("s1", ("t1", "cr"), ("u1",)),
                    Instruction("s2", ("u1", "t2"), ("u2",)),
                    Instruction("s3", ("u2", "b"), ("u3",)),
                    Instruction("ret", ("u3",), ()),
                ),
                succs=(),
            ),
        ),
    )


def fixed_register_program() -> Program:
    """Calling-convention style fixed registers.

    * the argument of ``syscall`` must arrive in r1;
    * its result is delivered in r0;
    * a live neighbor is forced into r0 elsewhere (collision potential).
    """
    return Program(
        entry="entry",
        params=("n",),
        blocks=(
            Block(
                "entry",
                instrs=(
                    Instruction("setup", ("n",), ("arg", "busy")),
                    Instruction(
                        "syscall",
                        ("arg",),
                        ("rc",),
                        fixed_use={"arg": 1},
                        fixed_def={"rc": 0},
                        clobbers=(1,),
                    ),
                    Instruction("fin", ("rc", "busy"), ("z",)),
                    Instruction("ret", ("z",), ()),
                ),
                succs=(),
            ),
        ),
    )


def loop_phi_program() -> Program:
    """A counted loop: header phi on both the entry edge and the back edge."""
    return Program(
        entry="entry",
        params=("n",),
        blocks=(
            Block(
                "entry",
                instrs=(Instruction("zero", (), ("z0",)),),
                succs=("head",),
            ),
            Block(
                "head",
                phis=(
                    Phi("i", {"entry": "z0", "body": "i2"}),
                    Phi("acc", {"entry": "z0", "body": "a2"}),
                ),
                instrs=(Instruction("cmp", ("i", "n"), ("cond",)),),
                succs=("body", "done"),
            ),
            Block(
                "body",
                instrs=(
                    Instruction("addacc", ("acc", "i"), ("a1",)),
                    Instruction("addi", ("i",), ("i2",)),
                    Instruction("mv", ("a1",), ("a2",)),
                ),
                succs=("head",),
            ),
            Block(
                "done",
                instrs=(Instruction("ret", ("acc",), ()),),
                succs=(),
            ),
        ),
    )


def high_pressure_program(n: int = 7) -> Program:
    """``n`` simultaneous long-lived values joined at the end.

    With K in 2..8, choosing ``n=7`` guarantees spills for every supported K.
    """
    assert 1 <= n <= 20
    defs = tuple(f"v{i}" for i in range(n))
    instrs = [Instruction(f"li{i}", (), (d,)) for i, d in enumerate(defs)]
    instrs.append(Instruction("merge0", (defs[0], defs[1]), ("acc1",)))
    for i in range(2, n):
        instrs.append(Instruction(f"merge{i}", (f"acc{i - 1}", defs[i]), (f"acc{i}",)))
    instrs.append(Instruction("ret", (f"acc{n - 1}",), ()))
    return Program(
        entry="entry",
        params=(),
        blocks=(Block("entry", instrs=tuple(instrs), succs=()),),
    )


def unsat_fixed_collision_program() -> Program:
    """Two simultaneously live values both forced into r0 -> provably unsat."""
    return Program(
        entry="entry",
        params=("a", "b"),
        blocks=(
            Block(
                "entry",
                instrs=(
                    Instruction(
                        "twin",
                        ("a", "b"),
                        ("x", "y"),
                        fixed_use={"a": 0, "b": 0},
                    ),
                    Instruction("ret", ("x", "y"), ()),
                ),
                succs=(),
            ),
        ),
    )


def unsat_fixed_clobber_program() -> Program:
    """A value forced into r1 that must survive a clobber of r1."""
    return Program(
        entry="entry",
        params=("a",),
        blocks=(
            Block(
                "entry",
                instrs=(
                    Instruction(
                        "call",
                        ("a",),
                        ("c",),
                        fixed_use={"a": 1},
                        clobbers=(1,),
                    ),
                    Instruction("ret", ("a", "c"), ()),
                ),
                succs=(),
            ),
        ),
    )
