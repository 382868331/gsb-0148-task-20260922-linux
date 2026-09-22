"""Phi lowering / critical edge / parallel-copy tests."""

import unittest

from regalloc import allocate
from regalloc.liveness import liveness_with_phis
from regalloc.lower import _sequential_parallel
from regalloc.samples import sample_critical_phi, sample_loop_call

from _reference import check_liveness_equations, simulate_parallel


class TestLivenessEquationReference(unittest.TestCase):
    def test_reports_satisfy_equations(self):
        for prog in (sample_critical_phi(), sample_loop_call()):
            rep = liveness_with_phis(prog)
            self.assertEqual(check_liveness_equations(prog, rep), [])


class TestParallelSequentialisation(unittest.TestCase):
    SCRATCH = 99

    def seq(self, actions):
        return _sequential_parallel([list(a) for a in actions], self.SCRATCH)

    def test_identity_copy_removed(self):
        mis = self.seq([("copy", 1, 1, -1)])
        self.assertEqual(mis, [])

    def test_simple_chain(self):
        actions = [("copy", 1, 0, -1), ("copy", 2, 1, -1)]
        mis = self.seq(actions)
        self.assertEqual(
            simulate_parallel(actions, mis, k=3, slots=set(), scratch_slot=self.SCRATCH),
            [],
        )
        # order must be 2<-1 before 1<-0 ... actually 1<-0 must not happen
        # first; ensure at least semantic correctness, and deterministically
        mis2 = self.seq(actions)
        self.assertEqual([(m.kind, m.dst, m.src) for m in mis],
                         [(m.kind, m.dst, m.src) for m in mis2])

    def test_two_cycle_uses_scratch_not_register(self):
        actions = [("copy", 0, 1, -1), ("copy", 1, 0, -1)]
        mis = self.seq(actions)
        self.assertEqual(
            simulate_parallel(actions, mis, k=2, slots=set(), scratch_slot=self.SCRATCH),
            [],
        )
        kinds = [m.kind for m in mis]
        self.assertIn("store", kinds)
        self.assertIn("load", kinds)

    def test_three_cycle(self):
        actions = [("copy", 0, 2, -1), ("copy", 1, 0, -1), ("copy", 2, 1, -1)]
        mis = self.seq(actions)
        self.assertEqual(
            simulate_parallel(actions, mis, k=3, slots=set(), scratch_slot=self.SCRATCH),
            [],
        )

    def test_cycles_and_stores_mixed(self):
        actions = [
            ("copy", 0, 1, -1),
            ("copy", 1, 0, -1),
            ("store", -1, 2, 7),
            ("copy", 3, 2, -1),
        ]
        mis = self.seq(actions)
        self.assertEqual(
            simulate_parallel(actions, mis, k=4, slots={7}, scratch_slot=self.SCRATCH),
            [],
        )

    def test_deterministic(self):
        actions = [("copy", 0, 1, -1), ("copy", 1, 0, -1), ("copy", 2, 3, -1)]
        a = [(m.kind, m.dst, m.src, m.slot) for m in self.seq(actions)]
        b = [(m.kind, m.dst, m.src, m.slot) for m in self.seq(actions)]
        self.assertEqual(a, b)


class TestCriticalEdgeSplit(unittest.TestCase):
    def test_split_blocks_inserted(self):
        prog = sample_critical_phi()
        alloc = allocate(prog, 3)
        ap = alloc.allocated_program
        splits = [b for b in ap.blocks if b.origin < 0]
        # edges 0->3 and 1->3 are critical; 2->3 is not
        edges = {tuple(b.split_edge) for b in splits}
        self.assertIn((0, 3), edges)
        self.assertIn((1, 3), edges)
        self.assertNotIn((2, 3), edges)
        for b in splits:
            self.assertEqual(len(b.succs), 1)
            self.assertFalse(any(mi.kind == "op" for mi in b.instrs))

    def test_split_preserves_reachability_and_terminals(self):
        prog = sample_critical_phi()
        alloc = allocate(prog, 3)
        ap = alloc.allocated_program
        # every split block is reachable from origin block 0
        seen = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            for s in ap.blocks[cur].succs:
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        self.assertEqual(seen, set(range(len(ap.blocks))))

    def test_self_loop_critical_edge_split(self):
        prog = sample_loop_call()
        alloc = allocate(prog, 2)
        ap = alloc.allocated_program
        edges = {tuple(b.split_edge) for b in ap.blocks if b.origin < 0}
        self.assertIn((1, 1), edges)


class TestLoweredIRShape(unittest.TestCase):
    def test_only_legal_kinds_and_regs(self):
        for prog in (sample_critical_phi(), sample_loop_call()):
            alloc = allocate(prog, 3)
            for b in alloc.allocated_program.blocks:
                for mi in b.instrs:
                    self.assertIn(mi.kind, {"op", "copy", "load", "store"})
                    for r in (*mi.reads, *mi.writes):
                        self.assertTrue(0 <= r < alloc.k)

    def test_each_uid_preserved_once(self):
        prog = sample_loop_call()
        alloc = allocate(prog, 2)
        uids = [mi.uid for b in alloc.allocated_program.blocks for mi in b.instrs
                if mi.kind == "op"]
        src_uids = [ins.uid for blk in prog.blocks for ins in blk.instrs]
        self.assertEqual(sorted(uids), sorted(src_uids))
