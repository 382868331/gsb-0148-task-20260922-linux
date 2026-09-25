"""Tiny concrete interpreters for before/after allocation equivalence.

Unlike :mod:`regalloc.symbolic` (provenance tokens, bounded path DFS), these
interpreters run a *single concrete execution* with real integer inputs over
a deliberately tiny opcode set and compare the numeric result the source
program computes against the one the lowered register/slot program computes.
Neither interpreter imports or calls the allocator core; they share only the
opcode semantics below, which are defined by the instruction ``note`` string:

    "const:V"  def = integer V
    "add"      def = a + b
    "sub"      def = a - b
    "mul"      def = a * b
    "lt"       def = 1 if a < b else 0
    "br3"      no def; reads (cond, sign): cond != 0 -> successor 0,
               else sign >= 0 -> successor 1, else successor 2
    "br2"      no def; reads (cond,): cond != 0 -> successor 0 else 1

Anything else raises: the fixture programs are fully enumerated, so an
unknown opcode is a test/demo authoring error, not a silent default.
"""

from __future__ import annotations

from dataclasses import dataclass


class ConcreteInterpError(RuntimeError):
    pass


def apply_op(note: str, inputs: tuple[int, ...]) -> tuple[int, ...] | str:
    """Return the def values, or ``"branch:<index>"`` for a branch op."""
    if note.startswith("const:"):
        return (int(note.split(":", 1)[1]),)
    if note == "add":
        return (inputs[0] + inputs[1],)
    if note == "sub":
        return (inputs[0] - inputs[1],)
    if note == "mul":
        return (inputs[0] * inputs[1],)
    if note == "lt":
        return (1 if inputs[0] < inputs[1] else 0,)
    if note == "br2":
        return f"branch:{0 if inputs[0] != 0 else 1}"
    if note == "br3":
        if inputs[0] != 0:
            idx = 0
        elif inputs[1] >= 0:
            idx = 1
        else:
            idx = 2
        return f"branch:{idx}"
    raise ConcreteInterpError(f"unknown opcode note {note!r}")


@dataclass
class ConcreteRun:
    exit_block: int
    values: dict
    block_trace: tuple[int, ...]
    uid_trace: tuple[int, ...]


def run_source(program, param_values: dict[int, int], *, max_steps: int = 100_000) -> ConcreteRun:
    """Execute the source :class:`Program` concretely to an exit block."""
    env: dict[int, int] = dict(param_values)
    cur = 0
    pred = -1
    block_trace: list[int] = []
    uid_trace: list[int] = []
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise ConcreteInterpError("source run exceeded step budget")
        b = program.blocks[cur]
        block_trace.append(cur)
        # phis are a simultaneous edge assignment: read every source from the
        # pre-update environment, then publish all results together
        phi_updates = {
            phi.def_var: env[dict(phi.sources)[pred]] for phi in b.phis
        }
        env.update(phi_updates)
        branch = None
        for ins in b.instrs:
            uid_trace.append(ins.uid)
            inputs = tuple(env[u] for u in ins.uses)
            out = apply_op(ins.note, inputs)
            if isinstance(out, str):
                if ins.defs:
                    raise ConcreteInterpError(
                        f"branch op uid {ins.uid} must not define a value"
                    )
                branch = int(out.split(":")[1])
                continue
            for d, val in zip(ins.defs, out):
                env[d] = val
        if not b.succs:
            return ConcreteRun(cur, dict(env), tuple(block_trace), tuple(uid_trace))
        if branch is None:
            nxt = b.succs[0]
        else:
            if branch >= len(b.succs):
                raise ConcreteInterpError(
                    f"block {cur}: branch chose successor index {branch}, "
                    f"only {len(b.succs)} successors"
                )
            nxt = b.succs[branch]
        pred, cur = cur, nxt


def run_allocated(program, allocation, param_values: dict[int, int],
                  *, max_steps: int = 100_000) -> ConcreteRun:
    """Execute the lowered register/slot program concretely to an exit block.

    The value of every source variable is recoverable at the end through its
    home register or spill slot, so the source-side result dict is rebuilt.
    """
    ap = allocation.allocated_program
    k = allocation.k
    homes: dict[int, tuple[str, int]] = {}
    for v, r in allocation.assignment.items():
        homes[v] = ("reg", r)
    for v, slot in allocation.slots.items():
        homes[v] = ("slot", slot)

    regs: dict[int, int] = {}
    slots: dict[int, int] = {}
    for v, val in param_values.items():
        kind, pos = homes[v]
        (regs if kind == "reg" else slots)[pos] = val

    # map source block index -> allocated block index, and allocated block
    # successor order follows source successor order (critical-edge split
    # blocks are substituted in place).
    origin_of: dict[int, int] = {}
    for ai, ab in enumerate(ap.blocks):
        if ab.origin >= 0:
            origin_of[ab.origin] = ai

    def execute_mi(mi) -> None:
        nonlocal branch
        if mi.kind == "copy":
            regs[mi.dst] = regs[mi.src]
        elif mi.kind == "load":
            regs[mi.dst] = slots[mi.slot]
        elif mi.kind == "store":
            slots[mi.slot] = regs[mi.src]
        elif mi.kind == "op":
            uid_trace.append(mi.uid)
            inputs = tuple(regs[r] for r in mi.reads)
            out = apply_op(mi.note, inputs)
            if isinstance(out, str):
                branch = int(out.split(":")[1])
            else:
                for r, val in zip(mi.writes, out):
                    regs[r] = val
        else:  # pragma: no cover
            raise ConcreteInterpError(f"unknown MI kind {mi.kind!r}")

    def follow_chain(ai: int) -> int:
        """Execute every pure edge-split block on the way, return next origin."""
        while ap.blocks[ai].origin < 0:
            if len(ap.blocks[ai].succs) != 1:
                raise ConcreteInterpError("split block with != 1 successor")
            for mi in ap.blocks[ai].instrs:
                execute_mi(mi)
            ai = ap.blocks[ai].succs[0]
        return ai

    cur = origin_of[0]
    block_trace: list[int] = []
    uid_trace: list[int] = []
    branch = None
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise ConcreteInterpError("allocated run exceeded step budget")
        ab = ap.blocks[cur]
        src_index = ab.origin
        block_trace.append(src_index)
        branch = None
        for mi in ab.instrs:
            execute_mi(mi)

        src_block = program.blocks[src_index]
        if not src_block.succs:
            if ab.succs:
                raise ConcreteInterpError("exit origin block has successors")
            break
        if branch is None:
            nxt = ab.succs[0]
        else:
            nxt = ab.succs[branch]
        cur = follow_chain(nxt)

    values: dict[int, int] = {}
    for v, (kind, pos) in homes.items():
        values[v] = regs[pos] if kind == "reg" else slots.get(pos)
    return ConcreteRun(block_trace[-1], values, tuple(block_trace), tuple(uid_trace))


def exit_live_vars(program, exit_block: int) -> set[int]:
    """Variables whose value is observable at the end of a run.

    Only these may be compared between the source and allocated runs: a
    variable that is dead at the exit has its home register legitimately
    reused by later definitions, so its final "value" is meaningless.
    """
    from .liveness import liveness_with_phis

    report = liveness_with_phis(program)
    live = set(report.live_in[exit_block])
    for ins in program.blocks[exit_block].instrs:
        live.update(ins.defs)
    return live
