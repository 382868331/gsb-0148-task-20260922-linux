"""Edge-sensitive liveness for format v2 programs.

Extends the v1 phi dataflow with two more kinds of edge action:

* **exception edges** (``Instr.except_to``): an instruction may transfer
  control to a handler *before* its defs are written.  The handler-side
  values must therefore be available at the excepting instruction with the
  instruction's defs excluded, and the edge's parallel copies / phi sources
  run on that transfer.
* **explicit parallel copies on edges** (``Block.edge_copies``): each
  ``(dst, src)`` pair acts like an edge-specific phi — ``dst`` is born on
  exactly that edge and ``src`` is consumed there.

As in v1, result sets are checkable independently: per-block live-in/out,
per-edge live sets, per-instruction before/after, plus
``instr_except_live`` (the union of exception-edge requirements at each
instruction) and ``edge_is_exception``.

Callee-save / clobber rules do not change liveness sets; they constrain
register placement and are handled by the allocator and the bit-level
checker.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Block, Program
from .liveness import predecessors


@dataclass(frozen=True)
class LivenessReportV2:
    live_in: tuple[frozenset[int], ...]
    live_out: tuple[frozenset[int], ...]
    edge_live: dict[tuple[int, int], frozenset[int]]
    instr_before: tuple[tuple[frozenset[int], ...], ...]
    instr_after: tuple[tuple[frozenset[int], ...], ...]
    # live values required on exception transfers at instruction (b, j)
    instr_except_live: tuple[tuple[frozenset[int], ...], ...]
    # edge (p, s) -> True if some instruction in p can transfer to s
    edge_is_exception: dict[tuple[int, int], bool]
    # global program points used for stack-slot interval construction:
    # ("instr", b, j) / ("edge-normal", p, s) / ("edge-exc", p, s, j) -> int
    points: dict[tuple, int]

    def as_jsonable(self) -> dict:
        return {
            "live_in": [sorted(s) for s in self.live_in],
            "live_out": [sorted(s) for s in self.live_out],
            "edge_live": {f"{p}->{s}": sorted(v) for (p, s), v in sorted(self.edge_live.items())},
            "instr_before": [[sorted(s) for s in row] for row in self.instr_before],
            "instr_after": [[sorted(s) for s in row] for row in self.instr_after],
            "instr_except_live": [[sorted(s) for s in row] for row in self.instr_except_live],
            "edge_is_exception": {f"{p}->{s}": v for (p, s), v in sorted(self.edge_is_exception.items())},
        }


def _edge_terms(program: Program):
    """Pre-compute per-edge phi/copy source and destination sets."""
    blocks = program.blocks
    phi_defs: list[set[int]] = [set() for _ in blocks]
    phi_src: dict[tuple[int, int], set[int]] = {}
    for s, b in enumerate(blocks):
        for phi in b.phis:
            phi_defs[s].add(phi.def_var)
            for p, src in phi.sources:
                phi_src.setdefault((p, s), set()).add(src)
    edge_dst: dict[tuple[int, int], set[int]] = {}
    edge_src: dict[tuple[int, int], set[int]] = {}
    for p, b in enumerate(blocks):
        for ec in b.edge_copies:
            key = (p, ec.dst_block)
            edge_dst.setdefault(key, set())
            edge_src.setdefault(key, set())
            for d, s in ec.pairs:
                edge_dst[key].add(d)
                edge_src[key].add(s)
    return phi_defs, phi_src, edge_dst, edge_src


def liveness_v2(program: Program) -> LivenessReportV2:
    assert program.is_v2()
    blocks = program.blocks
    n = len(blocks)
    phi_defs, phi_src, edge_dst, edge_src = _edge_terms(program)

    # exception edges out of each instruction: exc_targets[b][j]
    exc_targets: list[list[tuple[int, ...]]] = []
    exc_set: list[set[int]] = []
    edge_is_exception: dict[tuple[int, int], bool] = {}
    for bi, b in enumerate(blocks):
        row: list[tuple[int, ...]] = []
        excs: set[int] = set()
        for ins in b.instrs:
            row.append(tuple(ins.except_to))
            excs.update(ins.except_to)
        exc_targets.append(row)
        exc_set.append(excs)
        for s in b.succs:
            edge_is_exception[(bi, s)] = s in excs

    all_edge_dst: list[set[int]] = [set() for _ in range(n)]
    for (_, s), ds in edge_dst.items():
        all_edge_dst[s] |= ds

    # block summaries (upward exposed uses / killed), phi + edge dsts kill at
    # entry; exception transfers do not change body transfer summaries.
    up_exp: list[set[int]] = []
    killed: list[set[int]] = []
    for i, b in enumerate(blocks):
        killed_b = set(phi_defs[i]) | all_edge_dst[i]
        up_b: set[int] = set()
        for ins in b.instrs:
            up_b.update(v for v in ins.uses if v not in killed_b)
            killed_b.update(ins.defs)
        up_exp.append(up_b)
        killed.append(killed_b)

    # normal block-end successors: a successor that any instruction can
    # reach exceptionally is an exception-ONLY edge and receives no block-end
    # control flow (standard invoke representation).
    normal_succs: list[list[int]] = [
        [s for s in b.succs if s not in exc_set[i]] for i, b in enumerate(blocks)
    ]

    def edge_requirement(p: int, s: int, live_in_s: set[int]) -> set[int]:
        return ((set(live_in_s) - phi_defs[s] - edge_dst.get((p, s), set()))
                | phi_src.get((p, s), set())
                | edge_src.get((p, s), set()))

    live_in: list[set[int]] = [set() for _ in range(n)]
    live_out: list[set[int]] = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            new_out: set[int] = set()
            for s in normal_succs[i]:
                new_out |= edge_requirement(i, s, live_in[s])
            new_in = up_exp[i] | (new_out - killed[i])
            if new_out != live_out[i] or new_in != live_in[i]:
                live_out[i] = new_out
                live_in[i] = new_in
                changed = True

    edge_live: dict[tuple[int, int], frozenset[int]] = {}
    for i, b in enumerate(blocks):
        for s in b.succs:
            edge_live[(i, s)] = frozenset(edge_requirement(i, s, live_in[s]))

    # per-instruction backward scan; seed from normal successors only; the
    # exception requirement enters at the excepting instruction with that
    # instruction's defs excluded.
    instr_before: list[tuple[frozenset[int], ...]] = []
    instr_after: list[tuple[frozenset[int], ...]] = []
    instr_except: list[tuple[frozenset[int], ...]] = []
    for bi, b in enumerate(blocks):
        live: set[int] = set()
        for s in normal_succs[bi]:
            live |= set(edge_live[(bi, s)])
        befores: list[frozenset[int]] = []
        afters: list[frozenset[int]] = []
        excs: list[frozenset[int]] = []
        for j in range(len(b.instrs) - 1, -1, -1):
            ins = b.instrs[j]
            afters.append(frozenset(live))
            exc: set[int] = set()
            for t in ins.except_to:
                exc |= set(edge_live[(bi, t)])
            excs.append(frozenset(exc))
            live = ((live - set(ins.defs)) | set(ins.uses)
                    | (exc - set(ins.defs)))
            befores.append(frozenset(live))
        entry = frozenset(live - phi_defs[bi] - all_edge_dst[bi])
        assert entry == frozenset(live_in[bi]), (bi, entry, live_in[bi])
        instr_before.append(tuple(reversed(befores)))
        instr_after.append(tuple(reversed(afters)))
        instr_except.append(tuple(reversed(excs)))

    # global point numbering (deterministic, block/index order)
    points: dict[tuple, int] = {}
    counter = 0
    for bi, b in enumerate(blocks):
        for j in range(len(b.instrs) + 1):
            points[("instr", bi, j)] = counter
            counter += 1
        for s in b.succs:
            if edge_is_exception[(bi, s)]:
                # one point per excepting instruction targeting s
                for j, ins in enumerate(b.instrs):
                    if s in ins.except_to:
                        points[("edge-exc", bi, s, j)] = counter
                        counter += 1
            else:
                points[("edge-normal", bi, s)] = counter
                counter += 1

    return LivenessReportV2(
        live_in=tuple(frozenset(s) for s in live_in),
        live_out=tuple(frozenset(s) for s in live_out),
        edge_live=edge_live,
        instr_before=tuple(instr_before),
        instr_after=tuple(instr_after),
        instr_except_live=tuple(instr_except),
        edge_is_exception=edge_is_exception,
        points=points,
    )
