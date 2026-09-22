"""Paired symbolic-execution equivalence tests, including mutation tests.

The executor is independent of the allocator core.  Besides accepting every
real allocation, it must *reject* deliberately corrupted lowerings, proving
it has discriminating power and does not rubber-stamp results.
"""

import unittest

from regalloc import allocate
from regalloc.model import Block, Instr, PhiNode, Program
from regalloc.samples import (
    generate_pressure_program,
    sample_call_surviving,
    sample_critical_phi,
    sample_loop_call,
    sample_register_pressure,
)
from regalloc.symbolic import equivalence_check


class TestEquivalenceAccepts(unittest.TestCase):
    def test_all_samples(self):
        cases = [
            sample_call_surviving(),
            sample_critical_phi(),
            sample_loop_call(),
            sample_register_pressure(),
            generate_pressure_program(width=6),
        ]
        for prog in cases:
            for k in (2, 3, 4):
                with self.subTest(name=prog.name, k=k):
                    alloc = allocate(prog, k)
                    res = equivalence_check(prog, alloc, depth_bound=40, step_budget=8000)
                    self.assertTrue(res.equivalent, msg=res.failure)
                    self.assertGreaterEqual(res.ops_checked, 1)

    def test_spill_program_paths_covered(self):
        prog = sample_register_pressure()
        alloc = allocate(prog, 2)
        res = equivalence_check(prog, alloc)
        self.assertTrue(res.equivalent)
        self.assertEqual(res.paths_explored, 1)  # single straight-line path

    def test_loop_bounded(self):
        prog = sample_loop_call()
        alloc = allocate(prog, 2)
        res = equivalence_check(prog, alloc, depth_bound=10, step_budget=5000)
        self.assertTrue(res.equivalent)
        self.assertTrue(res.bound_reached)  # loop cannot be fully unrolled


class TestEquivalenceRejectsMutations(unittest.TestCase):
    def _alloc(self):
        return allocate(sample_register_pressure(), 3)

    def test_wrong_register_in_op_read_is_caught(self):
        alloc = self._alloc()
        # find an op with at least one read register and flip it to another
        for b in alloc.allocated_program.blocks:
            for mi in b.instrs:
                if mi.kind == "op" and mi.reads:
                    bad = (mi.reads[0] + 1) % alloc.k
                    object.__setattr__(mi, "reads", (bad,) + tuple(mi.reads[1:]))
                    res = equivalence_check(alloc.program, alloc)
                    self.assertFalse(res.equivalent)
                    self.assertIsNotNone(res.failure)
                    return
        self.fail("no mutable op found")

    def test_swallow_a_reload_is_caught(self):
        prog = sample_register_pressure()
        alloc = allocate(prog, 2)
        removed = False
        for b in alloc.allocated_program.blocks:
            keeps = []
            for mi in b.instrs:
                if not removed and mi.kind == "load":
                    removed = True  # drop this reload
                    continue
                keeps.append(mi)
            b.instrs = keeps
        self.assertTrue(removed)
        res = equivalence_check(prog, alloc)
        self.assertFalse(res.equivalent)

    def test_drop_phi_edge_copy_is_caught(self):
        # Diamond whose phi target is forced into r1 while sources arrive in
        # r0/r2 (pinned operands), so the critical-edge split blocks carry a
        # real copy r1 <- r0. Removing one copy corrupts that incoming path.
        b0 = Block(label="b0",
                   instrs=(Instr(0, uses=(), defs=(1,)),), succs=(1, 3))
        b1 = Block(label="b1",
                   instrs=(Instr(1, uses=(1,), defs=(2,), fixed={2: 2}),),
                   succs=(3, 2))
        b2 = Block(label="b2", instrs=(), succs=())
        b3 = Block(
            label="b3",
            instrs=(
                Instr(2, uses=(), defs=(4,), fixed={4: 2}),
                Instr(3, uses=(), defs=(6,), fixed={6: 0}),
                Instr(4, uses=(3, 4, 6), defs=(5,)),
            ),
            succs=(),
            phis=(PhiNode(3, ((0, 1), (1, 2))),),
        )
        prog = Program(blocks=(b0, b1, b2, b3))
        alloc = allocate(prog, 3)
        changed = False
        for b in alloc.allocated_program.blocks:
            if b.origin < 0 and any(mi.kind == "copy" for mi in b.instrs):
                b.instrs = [mi for mi in b.instrs if mi.kind != "copy"]
                changed = True
                break
        self.assertTrue(changed)
        res = equivalence_check(prog, alloc)
        self.assertFalse(res.equivalent)

    def test_stale_value_after_clobber_is_caught(self):
        prog = sample_call_surviving()
        alloc = allocate(prog, 3)
        # remove the clobber annotation: the executor must now either still
        # pass (value happened elsewhere) -- instead corrupt by redirecting
        # the surviving-value input to the clobbered register directly
        # Force v1 into r0 (the clobbered reg) by editing assignment/copies
        # is hard; simpler: append an op reading the call result reg after
        # the call but mark nothing. Do a concrete corruption: change the
        # call op's surviving operand register on uid2 to r0.
        for b in alloc.allocated_program.blocks:
            for mi in b.instrs:
                if mi.kind == "op" and mi.uid == 2:
                    # uid2 reads v1 and v2; make BOTH reads r0 (clobbered)
                    object.__setattr__(mi, "reads", (0, 0))
        res = equivalence_check(prog, alloc)
        self.assertFalse(res.equivalent)

    def test_wrong_slot_store_is_caught(self):
        prog = sample_register_pressure()
        alloc = allocate(prog, 2)
        for b in alloc.allocated_program.blocks:
            for mi in b.instrs:
                if mi.kind == "store":
                    object.__setattr__(mi, "slot", mi.slot + 100)
                    res = equivalence_check(prog, alloc)
                    self.assertFalse(res.equivalent)
                    return
        self.fail("no store found to corrupt")


class TestGeneratedProgramsEquivalence(unittest.TestCase):
    def test_several_seeds_and_k(self):
        for seed, width in ((1, 5), (42, 7), (20260922, 8)):
            prog = generate_pressure_program(seed=seed, width=width)
            for k in (2, 3, 4, 8):
                alloc = allocate(prog, k)
                res = equivalence_check(prog, alloc)
                self.assertTrue(res.equivalent, msg=f"seed={seed} k={k}: {res.failure}")
