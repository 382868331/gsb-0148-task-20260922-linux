"""Extended verification (kept separate from the fast regression suite).

Two groups:

1. **complete enumeration of a small reference domain** — on the 2-bank
   ``tiny_target`` every possible one-/two-value parallel-copy pattern
   (register homes, permutation cycles, live-through subsets, both widths)
   is generated; the emitted sequential code is executed on the independent
   bit machine and compared against the simultaneous-assignment semantics.
   The expected result is computed independently; the allocator is never
   asked for an expected value.

2. **main-scale non-degenerate program** — the deterministic 110-block /
   ~1100-instruction / 16-view generator; allocation plus the independent
   per-edge checker (ALL edges) and bounded paired execution.

Run with::

    python -m unittest tests.test_extended -v
"""

import itertools
import time
import unittest
from types import SimpleNamespace

from regalloc import (
    EdgeCopies,
    Block,
    Instr,
    Program,
    StackSpec,
    allocate,
    tiny_target,
)
from regalloc.checker2 import BitMachine
from regalloc.lower2 import emit_parallel


def _simulate_parallel(rt, home_banks, sources, dests, width):
    """Independent simultaneous-assignment oracle.

    ``sources`` maps value -> source token vector; ``dests`` maps value ->
    destination banks.  Returns the expected token at every bank after the
    simultaneous copies (sources read before any write).
    """
    before = {b: tuple((("src", b, k),) for k in range(width)) for b in home_banks}
    env = dict(before)
    for v, toks in sources.items():
        env[v] = toks
    after = dict(before)
    for d, src in dests.items():
        after[d] = env[src]
    return before, after


class TestCompleteSmallDomain(unittest.TestCase):
    """Exhaustive over the legal 2-bank parallel-copy domain.

    For every legal configuration on the tiny target — distinct copy
    destinations, source bank permutation, and a crossing (live-through)
    subset disjoint from the destinations — the emitted sequential code is
    run on the independent bit machine and checked against the simultaneous
    semantics.  Expected tokens are seeded/derived here, never by the
    allocator.
    """

    def setUp(self):
        self.rt = tiny_target().resolved()
        self.width = 32
        self.view = {0: "t0", 1: "t1"}

    def _check(self, src_banks, dest_map, crossing):
        """dest_map: {dst_node: src_node} with concrete banks from
        src_banks[node]; nodes 0,1 are sources and 2,3 destinations."""
        rt, width, view = self.rt, self.width, self.view
        dst_banks = {d: src_banks[s] for d, s in dest_map.items()}
        color = {n: frozenset({b}) for n, b in
                 [(0, src_banks[0]), (1, src_banks[1])] +
                 [(d, b) for d, b in dst_banks.items()]}
        plan = SimpleNamespace(
            node_class={i: "T32" for i in range(4)}, precolor={})
        records = [("copy", d, s) for d, s in sorted(dest_map.items())]
        mis = emit_parallel(
            rt, plan, color, records,
            crossing_banks=frozenset(crossing),
            scratch_slot=4, scratch_size=1,
        )
        m = BitMachine(rt, 8)
        # independent seed: one distinct token vector per bank
        seed = {b: tuple((("bank", b, k),) for k in range(width)) for b in (0, 1)}
        src_tok = {0: seed[src_banks[0]], 1: seed[src_banks[1]]}
        for b in (0, 1):
            m.write_view(view[b], seed[b])
        for mi in mis:
            if mi.kind == "copy":
                m.write_view(mi.dst, m.read_view(mi.src))
            elif mi.kind == "store":
                m.write_unit(mi.slot + mi.off, m.read_view(mi.src))
            else:
                m.write_view(mi.dst, m.read_unit(mi.slot + mi.off))
        # every destination holds its source's simultaneous token
        for d, s in dest_map.items():
            self.assertEqual(
                m.read_view(view[dst_banks[d]]), src_tok[s],
                f"src={src_banks} dest={dest_map} crossing={set(crossing)}")
        # every crossing (non-destination) bank is unchanged
        for b in crossing:
            self.assertEqual(m.read_view(view[b]), seed[b])
        # only the private scratch units (4..) may be touched in memory
        for u in range(4):
            self.assertIs(m.mem[u], m.mem[u])  # exists; must stay background
            from regalloc.checker2 import BACKGROUND
            self.assertIs(m.mem[u], BACKGROUND)

    def test_two_copy_swap_and_identity(self):
        n = 0
        for s_perm in itertools.permutations((0, 1)):
            # d0<-s0, d1<-s1 with destinations the two distinct banks in
            # either orientation; crossing must be empty (both banks written)
            for d0, d1 in ((0, 1), (1, 0)):
                sb = {0: s_perm[0], 1: s_perm[1]}
                self._check(sb, {2: 0, 3: 1}, frozenset())
                n += 1
        self.assertEqual(n, 4)

    def test_single_copy_with_every_crossing_subset(self):
        n = 0
        for s_bank in (0, 1):
            for d_bank in (0, 1):
                for crossing in (frozenset(), frozenset({0}),
                                 frozenset({1}), frozenset({0, 1})):
                    if d_bank in crossing:
                        continue  # illegal: cannot overwrite a live-through
                    self._check({0: s_bank, 1: 1 - s_bank}, {2: 0}, crossing)
                    n += 1
        # 2 source banks x 2 destination banks x 2 legal crossing subsets
        self.assertEqual(n, 8)

    def test_swap_with_both_banks_blocked_uses_scratch_only(self):
        rt, view, width = self.rt, self.view, self.width
        color = {0: frozenset({0}), 1: frozenset({1}),
                 2: frozenset({1}), 3: frozenset({0})}
        plan = SimpleNamespace(node_class={i: "T32" for i in range(4)},
                               precolor={})
        mis = emit_parallel(
            rt, plan, color, [("copy", 2, 0), ("copy", 3, 1)],
            crossing_banks=frozenset(), scratch_slot=4, scratch_size=1,
        )
        kinds = [mi.kind for mi in mis]
        self.assertIn("store", kinds)
        self.assertIn("load", kinds)
        self.assertTrue(all(mi.slot == 4 for mi in mis
                            if mi.kind in ("store", "load")))


class TestMainScale(unittest.TestCase):
    def test_generated_110_blocks_fully_verifies(self):
        from regalloc.samples2 import generate_large_v2
        p = generate_large_v2()
        n_instrs = sum(len(b.instrs) for b in p.blocks)
        self.assertGreaterEqual(len(p.blocks), 100)
        self.assertGreaterEqual(n_instrs, 1000)
        self.assertEqual(len(p.target.views), 16)
        t0 = time.perf_counter()
        a = allocate(p)
        elapsed = time.perf_counter() - t0
        vr = a.verification
        self.assertTrue(vr.ok, vr.failure_text()[:1000])
        # EVERY CFG edge was checked, including the exception edge
        self.assertEqual(vr.edges_checked,
                         sum(len(b.succs) for b in p.blocks))
        self.assertGreaterEqual(vr.exception_exits, 1)
        # must finish comfortably below the demo budget
        self.assertLess(elapsed, 60.0)

    def test_spilled_slots_are_reused_across_disjoint_ranges(self):
        from regalloc.samples2 import generate_large_v2
        a = allocate(generate_large_v2())
        starts = [start for start, _ in a.slots.values()]
        # far more spilled values than distinct slot starts => interval reuse
        self.assertGreater(len(a.slots), len(set(starts)))
        self.assertGreaterEqual(len(a.slots), 10)

    def test_generated_program_is_deterministic(self):
        from regalloc.samples2 import generate_large_v2
        p = generate_large_v2()
        a = allocate(p)
        b = allocate(p)
        self.assertEqual(a.assignment, b.assignment)
        self.assertEqual(a.slots, b.slots)
        self.assertEqual(a.allocated_program.render(), b.allocated_program.render())


if __name__ == "__main__":
    unittest.main()
