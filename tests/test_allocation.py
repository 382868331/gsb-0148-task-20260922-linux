"""Allocation tests.

The invariant checker here is independent of the allocator core: given only
the source program, the edge-sensitive liveness report and the resulting
assignment/slots, it verifies that no two simultaneously-live, non-spilled
values share a register and that call clobbers are respected.  It never
inspects the allocator's interference graph or plans.
"""

import unittest

from regalloc import allocate
from regalloc.allocator import exact_color, greedy_color
from regalloc.liveness import liveness_with_phis
from regalloc.model import Block, Instr, Program
from regalloc.samples import (
    generate_pressure_program,
    sample_call_surviving,
    sample_critical_phi,
    sample_loop_call,
    sample_register_pressure,
)
from regalloc import BudgetExhaustedError, InfeasibleProgramError


# --------------------------------------------------------------------------
# Independent invariant checker (does NOT use the allocator core)
# --------------------------------------------------------------------------

def independent_location_invariants(prog, alloc):
    """Return a list of human-readable violations (empty == all hold)."""
    rep = liveness_with_phis(prog)
    colors = alloc.assignment
    spilled = set(alloc.slots)
    violations = []

    def home(v):
        return None if v in spilled else colors.get(v)

    # before each instruction: distinct homes among live values
    for bi, b in enumerate(prog.blocks):
        for ii, ins in enumerate(b.instrs):
            for point_name, live in (
                ("before", rep.instr_before[bi][ii]),
                ("after", rep.instr_after[bi][ii]),
            ):
                seen: dict[int, int] = {}
                for v in live:
                    r = home(v)
                    if r is None:
                        continue
                    if r in seen and seen[r] != v:
                        violations.append(
                            f"b{bi} uid{ins.uid} {point_name}: v{v} and v{seen[r]} share r{r}"
                        )
                    seen[r] = v
            # a value surviving a call must not sit in a clobbered register
            survive = rep.instr_after[bi][ii] - set(ins.defs)
            for v in survive:
                r = home(v)
                if r is not None and r in ins.clobbers:
                    violations.append(
                        f"b{bi} uid{ins.uid}: v{v} survives call in clobbered r{r}"
                    )

    # every CFG edge: distinct homes
    for (a, b), live in rep.edge_live.items():
        seen: dict[int, int] = {}
        for v in live:
            r = home(v)
            if r is None:
                continue
            if r in seen and seen[r] != v:
                violations.append(f"edge {a}->{b}: v{v} and v{seen[r]} share r{r}")
            seen[r] = v
    return violations


PROGRAMS = {
    "call": sample_call_surviving(),
    "phi": sample_critical_phi(),
    "pressure": sample_register_pressure(),
    "loop": sample_loop_call(),
    "gen5": generate_pressure_program(width=5),
    "gen8": generate_pressure_program(seed=7, width=8),
    "gennoclob": generate_pressure_program(seed=3, width=6, clobber=False),
}


class TestLocationInvariants(unittest.TestCase):
    def test_all_samples_all_k(self):
        for name, prog in PROGRAMS.items():
            for k in range(2, 9):
                with self.subTest(program=name, k=k):
                    alloc = allocate(prog, k)
                    violations = independent_location_invariants(prog, alloc)
                    self.assertEqual(violations, [])
                    # registers used must be in range
                    for v, r in alloc.assignment.items():
                        self.assertGreaterEqual(r, 0)
                        self.assertLess(r, k)
                    # every variable is either assigned or spilled exactly once
                    all_vars = set(prog.params)
                    for b in prog.blocks:
                        for ins in b.instrs:
                            all_vars.update(ins.defs)
                        for phi in b.phis:
                            all_vars.add(phi.def_var)
                    self.assertEqual(
                        set(alloc.assignment) | set(alloc.slots), all_vars
                    )
                    self.assertTrue(
                        set(alloc.assignment).isdisjoint(alloc.slots)
                    )

    def test_spill_slots_indexed_by_variable_id(self):
        alloc = allocate(sample_register_pressure(), 2)
        self.assertTrue(alloc.slots)
        for v, slot in alloc.slots.items():
            self.assertEqual(slot, v)


class TestDeterminism(unittest.TestCase):
    def test_repeat_allocation_identical(self):
        prog = generate_pressure_program(width=6)
        a1 = allocate(prog, 2)
        a2 = allocate(prog, 2)
        self.assertEqual(a1.assignment, a2.assignment)
        self.assertEqual(a1.slots, a2.slots)
        self.assertEqual(
            a1.allocated_program.render(), a2.allocated_program.render()
        )
        self.assertEqual(a1.history, a2.history)


class TestColoringSearch(unittest.TestCase):
    def test_greedy_on_path(self):
        vp = _FakeVP(adj={0: {1}, 1: {0, 2}, 2: {1}}, precolor={})
        c = greedy_color(vp, 2)
        self.assertEqual(c[0], c[2])
        self.assertNotEqual(c[0], c[1])

    def test_exact_triangle_k2_is_proven_infeasible(self):
        vp = _FakeVP(adj={0: {1, 2}, 1: {0, 2}, 2: {0, 1}}, precolor={})
        status, color = exact_color(vp, 2, budget=1000)
        self.assertEqual(status, "infeasible")
        self.assertIsNone(color)

    def test_exact_triangle_k3_found(self):
        vp = _FakeVP(adj={0: {1, 2}, 1: {0, 2}, 2: {0, 1}}, precolor={})
        status, color = exact_color(vp, 3, budget=1000)
        self.assertEqual(status, "found")
        self.assertEqual(len({color[0], color[1], color[2]}), 3)

    def test_budget_distinct_from_infeasible(self):
        # K=3 coloring of a 4-clique: the very first branch cannot finish
        # within a 1-assignment budget, so feasibility is "unknown", not
        # disproven.
        adj = {i: {j for j in range(4) if j != i} for i in range(4)}
        vp = _FakeVP(adj=adj, precolor={})
        status, _ = exact_color(vp, 3, budget=1)
        self.assertEqual(status, "budget")

    def test_precolored_conflict_path(self):
        # two adjacent nodes forced into the same colour is infeasible
        vp = _FakeVP(
            adj={0: {1}, 1: {0}},
            precolor={0: 0, 1: 0},
        )
        status, _ = exact_color(vp, 2, budget=100)
        self.assertEqual(status, "infeasible")


class _FakeNode:
    pass


class _FakeVP:
    """Minimal stand-in for VirtualProgram accepted by the colorers."""

    def __init__(self, adj, precolor):
        self.adj = {v: set(ns) for v, ns in adj.items()}
        self.precolor = dict(precolor)


class TestSpillActuallyRelievesPressure(unittest.TestCase):
    def test_k2_requires_spills_and_succeeds(self):
        alloc = allocate(sample_register_pressure(), 2)
        self.assertGreaterEqual(len(alloc.slots), 1)
        self.assertLess(alloc.iterations, alloc.program.var_count() + 1)

    def test_more_registers_fewer_or_equal_spills(self):
        low = allocate(sample_register_pressure(), 2)
        high = allocate(sample_register_pressure(), 6)
        self.assertGreaterEqual(len(low.slots), len(high.slots))


class TestFixedRegistersMaterialised(unittest.TestCase):
    def _prog(self):
        return Program(
            blocks=(Block(
                label="e",
                instrs=(
                    Instr(uid=0, uses=(0, 1), defs=(2,), fixed={0: 0}, note="arg0 in r0"),
                    Instr(uid=1, uses=(2,), defs=(3,), fixed={3: 1}, note="ret in r1"),
                ),
            ),),
            params=(0, 1),
        )

    def test_fixed_use_and_def(self):
        prog = self._prog()
        alloc = allocate(prog, 3)
        mis = [mi for b in alloc.allocated_program.blocks for mi in b.instrs]
        ops = {mi.uid: mi for mi in mis if mi.kind == "op"}
        self.assertIn(0, ops[0].reads)   # r0 is read by uid0
        self.assertIn(1, ops[1].writes)  # uid1 writes r1

    def test_fixed_does_not_force_home_register(self):
        # v0's home register may differ from r0; copies materialise the pin
        prog = self._prog()
        alloc = allocate(prog, 3)
        mis = [mi for b in alloc.allocated_program.blocks for mi in b.instrs]
        # either v0 lives in r0 or an explicit copy brings it there
        if alloc.assignment.get(0) != 0:
            self.assertTrue(
                any(mi.kind == "copy" and mi.dst == 0 for mi in mis)
            )


class TestExceptionsDistinct(unittest.TestCase):
    def test_budget_class_is_separate(self):
        self.assertTrue(issubclass(BudgetExhaustedError, Exception))
        self.assertNotEqual(BudgetExhaustedError, InfeasibleProgramError)

    def test_budget_exhaustion_separate_from_infeasible(self):
        # Four simultaneously-live values make the greedy fail at K=3.
        # With the exact-search budget forced to 1, allocate must report
        # "budget exhausted" (feasibility unknown), NOT "proven infeasible".
        import regalloc.allocator as mod
        prog = sample_register_pressure()
        saved = mod.COLORING_BUDGET
        mod.COLORING_BUDGET = 1
        try:
            with self.assertRaises(BudgetExhaustedError):
                allocate(prog, 3)
        finally:
            mod.COLORING_BUDGET = saved
        # the same program with the real budget is simply allocated
        allocate(prog, 3)
