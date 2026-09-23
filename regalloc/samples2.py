"""Deterministic v2 sample machines/programs and a seeded scale generator.

Nothing here is random at run time: fixtures are constants and the
generator is driven by :class:`random.Random` with a fixed seed.
"""

from __future__ import annotations

import random

from .machine import Machine, RegView
from .model import Block, Instr, PhiNode, Program


# --------------------------------------------------------------------------
# machines
# --------------------------------------------------------------------------


def machine_torture() -> Machine:
    """Five groups, ten views: w/x pairs on g0-g2, s-class pair on g3-g4.

    Caller-saved: g0 (w0/x0).  Callee-saved: everything else.  The s class
    has exactly two views, so two s-class values pin them down completely.
    """
    views = []
    for g in range(3):
        views.append(RegView(f"w{g}", g, 0, 32, "w", callee_saved=(g > 0)))
        views.append(RegView(f"x{g}", g, 0, 64, "x", callee_saved=(g > 0)))
    views.append(RegView("s0", 3, 0, 32, "s", callee_saved=True))
    views.append(RegView("s1", 4, 0, 32, "s", callee_saved=True))
    return Machine(
        views=tuple(views),
        aliases={"w0": ("x0",), "x0": ("w0",), "w1": ("x1",), "x1": ("w1",),
                 "w2": ("x2",), "x2": ("w2",), "s0": (), "s1": ()},
        class_align={"w": 4, "x": 8, "s": 4},
        stack_align=8,
        name="torture",
    )


def machine_16() -> Machine:
    """Eight groups, sixteen views (w/x per group); g4-g7 callee-saved."""
    views = []
    aliases = {}
    for g in range(8):
        views.append(RegView(f"w{g}", g, 0, 32, "w", callee_saved=(g >= 4)))
        views.append(RegView(f"x{g}", g, 0, 64, "x", callee_saved=(g >= 4)))
        aliases[f"w{g}"] = (f"x{g}",)
        aliases[f"x{g}"] = (f"w{g}",)
    return Machine(
        views=tuple(views),
        aliases=aliases,
        class_align={"w": 4, "x": 8},
        stack_align=8,
        name="sixteen",
    )


def machine_reuse() -> Machine:
    """Two groups, four views, nothing callee-saved (slot-reuse demo)."""
    views = (
        RegView("w0", 0, 0, 32, "w"),
        RegView("x0", 0, 0, 64, "x"),
        RegView("w1", 1, 0, 32, "w"),
        RegView("x1", 1, 0, 64, "x"),
    )
    return Machine(
        views=views,
        aliases={"w0": ("x0",), "x0": ("w0",), "w1": ("x1",), "x1": ("w1",)},
        class_align={"w": 4, "x": 8},
        stack_align=8,
        name="reuse",
    )


# --------------------------------------------------------------------------
# the combined torture CFG
# --------------------------------------------------------------------------


def sample_v2_torture() -> Program:
    """One CFG with all v2 features at once.

    * exceptional edge ``b1 ~~> b3`` from a may_throw call, with a phi on
      the exceptional edge (parallel copy into the handler);
    * a double-width (x-class) value v3 live across the call and into the
      handler (callee-saved views only);
    * an early-clobber instruction (uid 12);
    * a phi swap cycle v4 <-> v5 on the loop back edge: on that edge five
      values are live (v4, v5 in the only two s-views; v3, v16 in x-views;
      v9 in a w-view), covering bits 0-31 of ALL five register groups, so
      no free temporary view exists and the cycle must be broken through
      the scratch stack slot;
    * genuine spills (v17/v20 must spill: three w-values survive the call
      but only one callee-saved w-view remains once v3 takes an x-view),
      so stack slots, alignment and slot-conflict checks are all active in
      the same allocation — the spilled pair is co-live inside the loop
      and their adjacent slots are checked against each other and against
      the scratch slot that breaks the copy cycle,

    Register budget note: v3 (x-class) is live throughout the loop, so at
    most two w-class values fit in registers at any loop point; the body
    is built to respect exactly that.
    """
    w = "w"
    var_classes = {
        0: w, 6: w, 8: w, 9: w, 10: w, 11: w, 13: w, 17: w, 20: w,
        3: "x", 21: "x", 15: "x", 16: "x",
        1: "s", 2: "s", 4: "s", 5: "s", 12: "s",
    }
    b0 = Block(
        label="entry",
        instrs=(
            Instr(uid=0, uses=(0,), defs=(1,), note="v1 = f(v0)"),
            Instr(uid=1, uses=(0,), defs=(2,), note="v2 = f(v0)"),
            Instr(uid=2, uses=(), defs=(3,), note="v3 = wide()"),
            Instr(uid=3, uses=(), defs=(21,), note="v21 = wide()"),
        ),
        succs=(1,),
    )
    b1 = Block(
        label="loop",
        instrs=(
            Instr(uid=10, uses=(4, 10, 3), defs=(6,), note="v6 = f(v4,v10,v3)"),
            Instr(uid=15, uses=(4, 15), defs=(17,), note="v17 = f(v4,v15)"),
            Instr(uid=16, uses=(5, 15), defs=(20,), note="v20 = f(v5,v15)"),
            Instr(uid=11, uses=(5, 3), defs=(), is_call=True, may_throw=True,
                  note="call may throw"),
            Instr(uid=12, uses=(6, 4, 5, 3), defs=(8,), early_clobber=(8,),
                  note="early-clobber def"),
            Instr(uid=13, uses=(8, 17, 4, 5, 3), defs=(9,), note="combine"),
            Instr(uid=14, uses=(9, 20, 3), defs=(16,), note="v16 = wide(...)"),
        ),
        succs=(1, 2),
        exc_succs=(3,),
        phis=(
            PhiNode(def_var=4, sources=((0, 1), (1, 5))),
            PhiNode(def_var=5, sources=((0, 2), (1, 4))),
            PhiNode(def_var=10, sources=((0, 0), (1, 9))),
            PhiNode(def_var=15, sources=((0, 21), (1, 16))),
        ),
    )
    b2 = Block(
        label="exit",
        instrs=(
            Instr(uid=20, uses=(4, 16, 9, 3), defs=(11,), note="result"),
            Instr(uid=21, uses=(11,), defs=(), note="sink"),
        ),
        succs=(),
    )
    b3 = Block(
        label="handler",
        instrs=(
            Instr(uid=30, uses=(12, 3), defs=(13,), note="handler use"),
            Instr(uid=31, uses=(13,), defs=(), note="sink"),
        ),
        succs=(),
        phis=(PhiNode(def_var=12, sources=((1, 5),)),),
    )
    return Program(
        blocks=(b0, b1, b2, b3),
        params=(0,),
        name="v2-torture",
        var_classes=var_classes,
    )


# --------------------------------------------------------------------------
# slot reuse + alignment
# --------------------------------------------------------------------------


def sample_slot_reuse() -> Program:
    """Two disjoint high-pressure generations on a 2-group machine.

    Generation 1 (v1,v2,v3) coexists, collapses to v5; generation 2
    (v6,v7,v8) coexists, collapses to v10.  With only two w-views one
    value per generation must spill, and the two spilled values have
    disjoint live ranges, so slot reuse is mandatory for a compact frame:
    they must end up at the same offset.  v0 is touched only once (low
    spill weight), so the allocator spills the generation temps rather
    than it.
    """
    var_classes = {i: "w" for i in range(13)}
    b0 = Block(
        label="main",
        instrs=(
            Instr(uid=0, uses=(), defs=(1,)),
            Instr(uid=1, uses=(), defs=(2,)),
            Instr(uid=2, uses=(), defs=(3,)),
            Instr(uid=3, uses=(1, 2), defs=(4,)),
            Instr(uid=4, uses=(4, 3), defs=(5,)),
            Instr(uid=5, uses=(), defs=(6,)),
            Instr(uid=6, uses=(), defs=(7,)),
            Instr(uid=7, uses=(), defs=(8,)),
            Instr(uid=8, uses=(6, 7), defs=(9,)),
            Instr(uid=9, uses=(9, 8), defs=(10,)),
            Instr(uid=10, uses=(5, 10), defs=(11,)),
            Instr(uid=11, uses=(11, 0), defs=(12,)),
            Instr(uid=12, uses=(12,), defs=()),
        ),
        succs=(),
    )
    return Program(
        blocks=(b0,), params=(0,), name="slot-reuse", var_classes=var_classes
    )


# --------------------------------------------------------------------------
# double-width value across a call (minimal, deterministic)
# --------------------------------------------------------------------------


def sample_wide_across_call() -> Program:
    """A 64-bit value must survive a call in a callee-saved x-view."""
    var_classes = {0: "w", 1: "x", 2: "w", 3: "w"}
    b0 = Block(
        label="entry",
        instrs=(
            Instr(uid=0, uses=(), defs=(1,), note="v1 = wide()"),
            Instr(uid=1, uses=(0, 1), defs=(2,), is_call=True, note="call"),
            Instr(uid=2, uses=(1, 2), defs=(3,), note="v1 used after call"),
        ),
        succs=(),
    )
    return Program(
        blocks=(b0,), params=(0,), name="wide-across-call",
        var_classes=var_classes,
    )


# --------------------------------------------------------------------------
# scale generator (>=100 blocks, >=1000 instrs, for the 16-view machine)
# --------------------------------------------------------------------------


def generate_scale_program(seed: int = 20260924, *, blocks: int = 110,
                           instrs_per_block: int = 10) -> Program:
    """Deterministic seeded scale program for the 16-view machine.

    Layout: block 0 defines 12 long-lived globals (6 w-class, 6 x-class)
    that are used across the whole program (they force genuine spilling
    against the 4+4 callee-saved views).  Blocks form a chain with
    periodic diamonds (with rejoin phis), calls, may_throw blocks with
    dedicated handler blocks (exceptional exits, handler phis), and two
    backward edges forming loops.  Only globals and same-block locals are
    referenced, so every use is dominated by its definition.
    """
    rng = random.Random(seed)
    n_handlers = 10
    n_normal = blocks - n_handlers - 1  # blocks 1..n_normal are normal
    handler0 = n_normal + 1             # first handler block index

    var_classes: dict[int, str] = {0: "w"}
    nxt = 1
    globals_w: list[int] = []
    globals_x: list[int] = []
    for _ in range(6):
        globals_w.append(nxt)
        var_classes[nxt] = "w"
        nxt += 1
    for _ in range(6):
        globals_x.append(nxt)
        var_classes[nxt] = "x"
        nxt += 1

    uid = 0
    blk_instrs: list[list[Instr]] = []
    blk_succs: list[tuple[int, ...]] = []
    blk_exc: list[tuple[int, ...]] = []
    blk_phis: list[list[PhiNode]] = []

    # block 0: define all globals
    entry: list[Instr] = []
    for g in globals_w + globals_x:
        entry.append(Instr(uid=uid, uses=(0,), defs=(g,), note="global"))
        uid += 1
    blk_instrs.append(entry)
    blk_succs.append((1,))
    blk_exc.append(())
    blk_phis.append([])

    throw_blocks: list[int] = []
    for i in range(1, n_normal + 1):
        instrs: list[Instr] = []
        locals_here: list[int] = []
        # a few local defs, each used later in the same block
        n_local = 3 + (i % 2)
        for j in range(n_local):
            cls = "x" if (i + j) % 3 == 0 else "w"
            v = nxt
            var_classes[v] = cls
            nxt += 1
            srcs = [rng.choice(globals_w)]
            if cls == "x":
                srcs = [rng.choice(globals_x)]
            early = (j == 0 and i % 5 == 0)
            instrs.append(
                Instr(uid=uid, uses=tuple(srcs), defs=(v,),
                      early_clobber=(v,) if early else (),
                      note="local")
            )
            uid += 1
            locals_here.append(v)
        # two rounds of uses of locals + globals
        for _ in range(2):
            for j, v in enumerate(locals_here):
                g = rng.choice(globals_x if var_classes[v] == "x" else globals_w)
                out = nxt
                var_classes[out] = var_classes[v]
                nxt += 1
                instrs.append(Instr(uid=uid, uses=(v, g), defs=(out,), note="mix"))
                uid += 1
                locals_here[j] = out
        # call every 7th block
        is_call = i % 7 == 3
        may_throw = i % 9 == 4
        if is_call or may_throw:
            g = rng.choice(globals_w)
            out = nxt
            var_classes[out] = "w"
            nxt += 1
            instrs.append(
                Instr(uid=uid, uses=(g, locals_here[0]), defs=(out,),
                      is_call=True, may_throw=may_throw, note="call")
            )
            uid += 1
        # pin one operand to a view every 11th block
        if i % 11 == 6 and locals_here:
            v = locals_here[-1]
            cls = var_classes[v]
            view = ("w5" if cls == "w" else "x5")
            out = nxt
            var_classes[out] = cls
            nxt += 1
            instrs.append(
                Instr(uid=uid, uses=(v,), defs=(out,), fixed_views={v: view},
                      note="pinned")
            )
            uid += 1
        # sink some locals so they die here
        if locals_here:
            instrs.append(Instr(uid=uid, uses=tuple(locals_here[:2]), defs=(),
                                note="sink"))
            uid += 1

        # successors: chain, diamonds, loops, exceptional edges
        nxt_blk = i + 1 if i < n_normal else 0
        succs: tuple[int, ...]
        exc: tuple[int, ...] = ()
        if i == n_normal:
            succs = ()  # normal exit at the end of the chain
        elif i % 10 == 5 and i + 2 <= n_normal:
            succs = (i + 1, i + 2)  # diamond branch
        elif i % 10 == 6 and i + 1 <= n_normal:
            succs = (i + 2,)  # left arm rejoins
        elif i == 45:
            succs = (i + 1, 20)  # backward edge (loop)
        elif i == 80:
            succs = (i + 1, 55)  # backward edge (loop)
        else:
            succs = (nxt_blk,) if nxt_blk else ()
        if may_throw:
            exc = (handler0 + (i % n_handlers),)
            throw_blocks.append(i)
        blk_instrs.append(instrs)
        blk_succs.append(succs)
        blk_exc.append(exc)
        blk_phis.append([])

    # rejoin phis: blocks 8, 18, 28, ... (i % 10 == 8) have preds 6,7 mod 10
    for i in range(8, n_normal + 1, 10):
        v = nxt
        var_classes[v] = "w"
        nxt += 1
        g1, g2 = rng.choice(globals_w), rng.choice(globals_w)
        blk_phis[i] = [PhiNode(def_var=v, sources=((i - 2, g1), (i - 1, g2)))]
        blk_instrs[i].append(Instr(uid=uid, uses=(v,), defs=(), note="phi use"))
        uid += 1

    # handler blocks: phi over the exceptional predecessors, then exit
    for h in range(n_handlers):
        hb = handler0 + h
        preds = [i for i in throw_blocks if blk_exc[i] == (hb,)]
        v = nxt
        var_classes[v] = "w"
        nxt += 1
        srcs = tuple((p, rng.choice(globals_w)) for p in preds)
        instrs = [
            Instr(uid=uid, uses=(v, rng.choice(globals_x)), defs=(nxt,),
                  note="handler"),
        ]
        var_classes[nxt] = "w"
        nxt += 1
        uid += 1
        instrs.append(Instr(uid=uid, uses=(nxt - 1,), defs=(), note="sink"))
        uid += 1
        blk_instrs.append(instrs)
        blk_succs.append(())
        blk_exc.append(())
        blk_phis.append([PhiNode(def_var=v, sources=srcs)])

    blocks_t = tuple(
        Block(
            label=(f"h{i - handler0}" if i >= handler0 else f"b{i}"),
            instrs=tuple(blk_instrs[i]),
            succs=blk_succs[i],
            exc_succs=blk_exc[i],
            phis=tuple(blk_phis[i]),
        )
        for i in range(len(blk_instrs))
    )
    return Program(
        blocks=blocks_t,
        params=(0,),
        name=f"scale-{seed}",
        var_classes=var_classes,
    )
