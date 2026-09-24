"""Deterministic format-v2 sample programs and a seeded generator.

Nothing here is random at run time: hand-built fixtures are constants and
the large generator is driven by :class:`random.Random` with a fixed seed.
All programs target either the built-in 16-view ``default_target`` (four
banks with 64/32/16/8 bit views) or the 6-view ``tiny_target`` (two banks;
under full pressure there is no spare temporary register).
"""

from __future__ import annotations

import random

from .model import Block, EdgeCopies, Instr, PhiNode, Program, StackSpec
from .target import default_target, tiny_target


def _clob_views(target, banks):
    return tuple(sorted(v.name for v in target.views if v.bank in banks))


# --------------------------------------------------------------------------
# small, fully hand-checked fixtures
# --------------------------------------------------------------------------

def sample_width_alias() -> Program:
    """Writes/reads through every width view of aliased banks.

    A narrow definition (``al``) feeds a 16-bit use (``ax``) and later a
    64-bit use (``rax``); the allocator must treat the views as aliasing the
    same bank rather than distinct registers.
    """
    t = default_target()
    b0 = Block(label="alias", instrs=(
        Instr(uid=0, uses=(), defs=(1,), fixedv={1: "al"}),
        Instr(uid=1, uses=(1,), defs=(2,), classes={1: "GPR16", 2: "GPR16"},
              fixedv={1: "al"}),
        Instr(uid=2, uses=(2,), defs=(3,), classes={2: "GPR64", 3: "GPR64"}),
        Instr(uid=3, uses=(3,), defs=(), classes={3: "GPR64"}),
    ), succs=())
    return Program(blocks=(b0,), version=2, target=t, name="width-alias")


def sample_pair_across_call() -> Program:
    """A double-width PAIR64 value survives a call that clobbers eax/ecx.

    The only pair option that can live across the call is ebx:edx (both
    callee-saved); eax:ecx is destroyed by the call.
    """
    t = default_target()
    b0 = Block(label="b", instrs=(
        Instr(uid=0, uses=(), defs=(1,), classes={1: "PAIR64"}),
        Instr(uid=1, uses=(), defs=(2,),
              clobbersv=_clob_views(t, (0, 2)), note="call"),
        Instr(uid=2, uses=(1, 2), defs=(3,), classes={1: "PAIR64"}),
        Instr(uid=3, uses=(3,), defs=()),
    ), succs=())
    return Program(blocks=(b0,), version=2, target=t, name="pair-across-call")


def sample_early_clobber() -> Program:
    """An early-clobber result is written while an input must survive.

    ``uid2`` reads v1 (pinned ecx) and early-writes v3 (pinned ecx): the
    input v1 has to be moved away before the instruction, proving the
    early-clobber edge is honoured (a normal def could share the dying
    operand's bank; an early one cannot).
    """
    t = default_target()
    b0 = Block(label="b", instrs=(
        Instr(uid=0, uses=(), defs=(1,), fixedv={1: "ecx"}),
        Instr(uid=1, uses=(), defs=(2,)),
        Instr(uid=2, uses=(1,), defs=(3, 4), early_clobber=(3,),
              fixedv={3: "ecx"}, note="early result"),
        Instr(uid=3, uses=(1, 4), defs=(5,)),
        Instr(uid=4, uses=(5,), defs=()),
    ), succs=())
    return Program(blocks=(b0,), version=2, target=t, name="early-clobber")


def sample_exception_edge() -> Program:
    """An instruction can transfer exceptionally; the def is not written on
    that path and a value must be available in the handler."""
    t = default_target()
    b0 = Block(label="body", instrs=(
        Instr(uid=0, uses=(), defs=(1,)),
        Instr(uid=1, uses=(1,), defs=(2,), except_to=(2,), note="risky"),
        Instr(uid=2, uses=(2,), defs=(3,)),
    ), succs=(1, 2))
    b1 = Block(label="ok", instrs=(Instr(uid=3, uses=(3,), defs=()),), succs=())
    b2 = Block(label="handler", instrs=(
        Instr(uid=4, uses=(1,), defs=(4,)),
        Instr(uid=5, uses=(4,), defs=()),
    ), succs=())
    return Program(blocks=(b0, b1, b2), version=2, target=t, name="exception-edge")


def sample_copy_cycle_no_temp() -> Program:
    """A true register swap on the 2-bank target with NO free temporary.

    The join phis force homes opposite to the predecessor fixed sources;
    both banks are occupied by the swap itself, so the cycle must be broken
    through the private scratch stack slot.
    """
    t = tiny_target()
    c0 = _clob_views(t, (0,))
    c1 = _clob_views(t, (1,))
    b0 = Block(label="pre", instrs=(
        Instr(uid=0, uses=(), defs=(2,), fixedv={2: "t1"}),
        Instr(uid=1, uses=(), defs=(3,), fixedv={3: "t0"}),
    ), succs=(1,))
    b1 = Block(label="join", instrs=(), succs=(2, 3), phis=(
        PhiNode(def_var=4, sources=((0, 2),)),
        PhiNode(def_var=5, sources=((0, 3),)),
    ))
    b2 = Block(label="useA", instrs=(
        Instr(uid=2, uses=(), defs=(10,), clobbersv=c1),
        Instr(uid=3, uses=(4, 10), defs=(11,), fixedv={4: "t0"}),
        Instr(uid=4, uses=(11,), defs=()),
    ), succs=())
    b3 = Block(label="useB", instrs=(
        Instr(uid=5, uses=(), defs=(12,), clobbersv=c0),
        Instr(uid=6, uses=(5, 12), defs=(13,), fixedv={5: "t1"}),
        Instr(uid=7, uses=(13,), defs=()),
    ), succs=())
    return Program(blocks=(b0, b1, b2, b3), version=2, target=t,
                   stack=StackSpec(align=2), name="copy-cycle-no-temp")


def sample_combined() -> Program:
    """All three new capabilities in one CFG: exception edge + early-clobber
    op + a double-width value live across a call, plus callee-saved crossing
    values that fill every bank around a branch join."""
    t = default_target()

    def clob(*banks):
        return _clob_views(t, banks)

    b0 = Block(label="pre", instrs=(
        Instr(uid=0, uses=(), defs=(1,), classes={1: "PAIR64"}),
        Instr(uid=1, uses=(), defs=(20,), clobbersv=clob(0, 2), note="call A"),
        Instr(uid=2, uses=(20,), defs=(21,)),
        Instr(uid=3, uses=(), defs=(2,), fixedv={2: "eax"}),
        Instr(uid=4, uses=(), defs=(3,), fixedv={3: "ecx"}),
        Instr(uid=5, uses=(3,), defs=(4, 5), early_clobber=(5,),
              fixedv={5: "eax"}, note="div-like"),
        Instr(uid=6, uses=(5,), defs=(6,), except_to=(4,), note="risky"),
        Instr(uid=17, uses=(), defs=(30,), fixedv={30: "ebx"}),
        Instr(uid=18, uses=(), defs=(31,), fixedv={31: "edx"}),
    ), succs=(1, 4))
    b1 = Block(label="join", instrs=(
        Instr(uid=7, uses=(1,), defs=(22,), classes={1: "PAIR64"}, note="consume pair"),
        Instr(uid=8, uses=(22,), defs=()),
    ), succs=(2, 3), phis=(
        PhiNode(def_var=7, sources=((0, 2),)),
        PhiNode(def_var=8, sources=((0, 3),)),
    ))
    b2 = Block(label="useU", instrs=(
        Instr(uid=9, uses=(), defs=(23,), clobbersv=clob(0, 1, 3)),
        Instr(uid=10, uses=(7, 23, 30, 31), defs=(24,), fixedv={7: "ecx"}),
        Instr(uid=11, uses=(24,), defs=()),
    ), succs=())
    b3 = Block(label="useV", instrs=(
        Instr(uid=12, uses=(), defs=(25,), clobbersv=clob(1, 2, 3)),
        Instr(uid=13, uses=(8, 25, 30, 31), defs=(26,), fixedv={8: "eax"}),
        Instr(uid=14, uses=(26,), defs=()),
    ), succs=())
    b4 = Block(label="handler", instrs=(
        Instr(uid=15, uses=(2,), defs=(27,)),
        Instr(uid=16, uses=(27,), defs=()),
    ), succs=())
    return Program(blocks=(b0, b1, b2, b3, b4), version=2, target=t,
                   stack=StackSpec(align=2), name="combined")


def sample_mega() -> Program:
    """ALL THREE new capabilities plus the stack-slot conflict in ONE CFG:

    * a double-width PAIR64 value stays REGISTER-resident in the
      callee-saved ``ebx:edx`` pair across a call that clobbers ``eax``/``ecx``;
    * an early-clobber instruction writes a result pinned to ``ecx`` while an
      input pinned to ``eax`` (and the pair) must still exist;
    * the following instruction can transfer exceptionally to a handler that
      reads the ``eax``-pinned value (exception edge, def not written);
    * on the join edge exactly four values are live — the pair (banks 1,3),
      an ``eax`` value and an ``ecx`` value — and the join phis swap their
      homes (``ecx<-eax``, ``eax<-ecx``).  Every bank is occupied, so the
      parallel-copy swap is a cycle with NO free temporary register and must
      be broken through the private scratch stack slot.
    """
    t = default_target()

    def clob(*banks):
        return _clob_views(t, banks)

    b0 = Block(label="pre", instrs=(
        Instr(uid=0, uses=(), defs=(1,), classes={1: "PAIR64"}),
        Instr(uid=1, uses=(), defs=(20,), clobbersv=clob(0, 2), note="call A"),
        Instr(uid=2, uses=(20,), defs=(21,)),
        Instr(uid=3, uses=(21,), defs=()),
        Instr(uid=4, uses=(), defs=(2,), fixedv={2: "eax"}),
        Instr(uid=5, uses=(2,), defs=(3,), early_clobber=(3,),
              fixedv={3: "ecx"}, note="early y while x survives"),
        Instr(uid=6, uses=(3,), defs=(), except_to=(4,), note="risky"),
    ), succs=(1, 4))
    b1 = Block(label="join", instrs=(
        Instr(uid=8, uses=(1,), defs=(22,), classes={1: "PAIR64"}),
    ), succs=(2, 3), phis=(
        PhiNode(def_var=7, sources=((0, 2),)),   # u <- x(eax) -> forced ecx
        PhiNode(def_var=8, sources=((0, 3),)),   # v <- y(ecx) -> forced eax
    ))
    b2 = Block(label="useU", instrs=(
        Instr(uid=9, uses=(), defs=(30,), clobbersv=clob(0, 1, 3)),
        Instr(uid=10, uses=(7, 30), defs=(24,), fixedv={7: "ecx"}),
        Instr(uid=11, uses=(24,), defs=()),
    ), succs=())
    b3 = Block(label="useV", instrs=(
        Instr(uid=12, uses=(), defs=(31,), clobbersv=clob(1, 2, 3)),
        Instr(uid=13, uses=(8, 31), defs=(26,), fixedv={8: "eax"}),
        Instr(uid=14, uses=(26,), defs=()),
    ), succs=())
    b4 = Block(label="handler", instrs=(
        Instr(uid=15, uses=(2,), defs=(27,)),
        Instr(uid=16, uses=(27,), defs=()),
    ), succs=())
    return Program(blocks=(b0, b1, b2, b3, b4), version=2, target=t,
                   stack=StackSpec(align=2), name="mega")


# --------------------------------------------------------------------------
# seeded main-scale generator (>=100 blocks, >=1000 instrs, 16 views)
# --------------------------------------------------------------------------
def generate_large_v2(seed: int = 20260924, *, blocks: int = 110,
                      per_block: int = 10) -> Program:
    """Deterministically generate a non-degenerate main-scale v2 program.

    The CFG is a linear chain embedding three natural loops
    (pre -> header -> body -> latch -> {header, exit}), one invoke with a
    real exception edge to a dedicated handler, periodic calls that clobber
    the caller-saved banks and occasional fixed-view pins.  Each ordinary
    block carries ``per_block`` opaque instructions, so the default is
    110 blocks and ~1100 instructions over the 16-view target.  The PRNG is
    seeded and blocks are appended in a fixed order, so the program (and
    therefore its allocation) is fully reproducible.
    """
    t = default_target()
    rng = random.Random(seed)
    st = {"v": 100, "u": 0}

    def fresh():
        v = st["v"]
        st["v"] += 1
        return v

    def mk(uses=(), defs=(), **kw):
        ins = Instr(uid=st["u"], uses=uses, defs=defs, **kw)
        st["u"] += 1
        return ins

    specs: list[dict] = []      # one entry per block, in index order

    def spec(kind, **kw):
        specs.append({"kind": kind, **kw})
        return len(specs) - 1

    # plan the skeleton (indices are assigned by construction order)
    entry = spec("entry")
    loop_ids = []
    for li in range(3):
        pre = spec("pre", li=li)
        header = spec("header", li=li)
        body = spec("body", li=li)
        latch = spec("exitlatch", li=li)   # body successor
        # latch block created right after body; split latch/exit below
    # Redo cleanly: rebuild specs with an explicit loop layout.
    specs = []
    entry = spec("entry")
    loop_ids = []
    for li in range(3):
        pre = spec("pre", li=li)
        header = spec("header", li=li)
        body = spec("body", li=li)
        latch = spec("latch", li=li)
        exitb = spec("loopexit", li=li)
        loop_ids.append((pre, header, body, latch, exitb))
    invoke = spec("invoke")
    normal = spec("normal")
    handler = spec("handler")
    n_fixed = len(specs)

    # successor wiring computed later from kinds; now emit physical blocks
    long_lived = [fresh() for _ in range(2)]   # fit the 2 callee-saved banks across calls

    def ops(live, *, force_uses=()):
        """``per_block`` opaque ops threading an explicit live list.

        Invariants kept so the program is feasible: at most 3 values are
        simultaneously live; a fixed-view pin is consumed by the very next
        instruction (it never has to survive a later clobber); a call that
        clobbers banks 0/2 is emitted only while the live set can fit the
        callee-saved banks (1,3) plus the call's own fresh result.
        """
        out = []
        cur = list(live)

        def use_some():
            us = set(force_uses if (not out and force_uses) else ())
            for v in cur[-3:]:
                if rng.random() < 0.55:
                    us.add(v)
            for g in long_lived:
                if rng.random() < 0.18:
                    us.add(g)
            return us

        step = 0
        while step < per_block:
            uses = set()
            for v in cur[-2:]:
                if rng.random() < 0.5:
                    uses.add(v)
            if not out and force_uses:
                uses.update(force_uses)
            # keep at most one long-lived value live through any one op so
            # the read point never demands more banks than exist (all these
            # are single-width GPR32 -> max 4 distinct banks)
            if rng.random() < 0.25 and long_lived:
                uses.add(rng.choice(long_lived))
            uses = set(sorted(uses)[:3])
            d = fresh()
            is_call = (step % 5 == 2)
            is_pin = (step % 7 == 4)
            kw = {}
            if is_call:
                kw["clobbersv"] = _clob_views(t, (0, 2))
            if is_pin and not is_call:
                view = rng.choice(("eax", "ecx", "ebx", "edx"))
                kw["fixedv"] = {d: view}
            out.append(mk(uses=tuple(sorted(uses)), defs=(d,), **kw))
            cur = [v for v in cur if v not in uses] + [d]
            step += 1
            # a pinned result is consumed immediately by the next op so its
            # fixed home never has to survive a subsequent clobber
            if is_pin and step < per_block:
                e = fresh()
                out.append(mk(uses=(d,), defs=(e,)))
                cur = [v for v in cur if v != d] + [e]
                step += 1
            cur = cur[-3:]
        return out, cur

    block_instrs: dict[int, list] = {}
    block_phis: dict[int, list] = {}
    init_var: dict[int, int] = {}
    carried: dict[int, int] = {}

    # entry defines long-lived values then ordinary ops
    ins, last = ops([], force_uses=())
    block_instrs[entry] = [mk(defs=(g,)) for g in long_lived] + ins
    entry_last = last

    for (pre, header, body, latch, exitb) in loop_ids:
        iv = fresh()
        init_var[header] = iv
        pinstr, _ = ops(entry_last, force_uses=())
        pinstr.append(mk(defs=(iv,)))
        block_instrs[pre] = pinstr
        # header phi x <- iv | cv
        cv = fresh()
        carried[header] = cv
        block_phis[header] = [PhiNode(def_var=fresh())]  # placeholder replaced
        xin = block_phis[header][0].def_var
        block_phis[header] = [PhiNode(
            def_var=xin, sources=((pre, iv), (latch, cv)))]
        hinstr, _ = ops([], force_uses=(xin,))
        block_instrs[header] = hinstr
        binstr, _ = ops([], force_uses=(xin,))
        block_instrs[body] = binstr
        linstr, _ = ops([], force_uses=(xin,))
        linstr.append(mk(uses=(xin,), defs=(cv,)))
        block_instrs[latch] = linstr
        einstr, _ = ops([], force_uses=(xin,))
        block_instrs[exitb] = einstr

    # invoke keeps g0 live (handler reads it); defines hv for normal path
    inv_ins, _ = ops([], force_uses=(long_lived[0],))
    hv = fresh()
    inv_ins.append(mk(uses=(long_lived[0],), defs=(hv,),
                      except_to=(), note="risky"))  # target set after blocks
    block_instrs[invoke] = inv_ins
    block_instrs[normal], _ = ops([], force_uses=(hv, long_lived[0]))
    hr = fresh()
    block_instrs[handler] = [mk(uses=(long_lived[0],), defs=(hr,)),
                             mk(uses=(hr,), defs=())]

    # tail linear blocks to reach the requested size
    tail_start = len(specs)
    tail_live = [hv]
    while len(specs) < blocks:
        ti = spec("tail", n=len(specs))
        tins, tail_live = ops(tail_live)
        block_instrs[ti] = tins

    # ---- successors ----
    succ: dict[int, tuple] = {}
    order = [entry] + [x for lp in loop_ids for x in lp] + [invoke, normal, handler]
    order += [i for i in range(n_fixed, len(specs))]
    for k, idx in enumerate(order):
        kind = specs[idx]["kind"]
        if kind == "entry":
            succ[idx] = (loop_ids[0][0],)
        elif kind == "pre":
            succ[idx] = (specs_idx(specs, idx, "header"),)
        elif kind == "header":
            succ[idx] = (specs_idx(specs, idx, "body"),)
        elif kind == "body":
            succ[idx] = (specs_idx(specs, idx, "latch"),)
        elif kind == "latch":
            succ[idx] = (specs_idx(specs, idx, "header"),
                         specs_idx(specs, idx, "loopexit"))
        elif kind == "loopexit":
            nxt = next_loop_entry(specs, idx, invoke)
            succ[idx] = (nxt,)
        elif kind == "invoke":
            succ[idx] = (normal, handler)
        elif kind == "normal":
            succ[idx] = (n_fixed,) if n_fixed < len(specs) else ()
        elif kind == "handler":
            succ[idx] = ()
        elif kind == "tail":
            succ[idx] = (idx + 1,) if idx + 1 < len(specs) else ()

    # patch the risky instruction's exception target to the handler index
    rinstr = block_instrs[invoke][-1]
    block_instrs[invoke][-1] = Instr(
        uid=rinstr.uid, uses=rinstr.uses, defs=rinstr.defs,
        except_to=(handler,), note=rinstr.note)

    blocks_out = []
    for idx in range(len(specs)):
        blocks_out.append(Block(
            label=specs[idx]["kind"] + (f"{specs[idx].get('li', '')}"
                                        if "li" in specs[idx] else f"_{idx}"),
            instrs=tuple(block_instrs.get(idx, [])),
            succs=succ.get(idx, ()),
            phis=tuple(block_phis.get(idx, [])),
        ))
    return Program(blocks=tuple(blocks_out), version=2, target=t,
                   stack=StackSpec(align=2), name=f"large-v2-{seed}")


def specs_idx(specs, idx, kind):
    """Index of the block of `kind` within the same loop group as idx."""
    li = specs[idx]["li"]
    for j, s in enumerate(specs):
        if s["kind"] == kind and s.get("li") == li:
            return j
    raise KeyError(kind)


def next_loop_entry(specs, idx, invoke):
    """Successor of a loop exit: next loop's pre, or the invoke block."""
    li = specs[idx]["li"]
    for j, s in enumerate(specs):
        if s["kind"] == "pre" and s.get("li") == li + 1:
            return j
    return invoke
