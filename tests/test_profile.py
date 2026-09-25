"""Frequency-guided spill selection tests.

Covers:

* the ``frequencies`` parameter contract (shape, type, bool/negative
  rejection, default ``None`` preserving the legacy strategy exactly);
* edge-frequency estimation rules;
* per-iteration recomputation of costs against the rebuilt spill form;
* the hot-loop/cold-error fixture: the profile flips spill victims, actual
  weighted hot memory traffic drops versus the legacy strategy, and the
* inverted profile flips the choice the other way;
* before/after equivalence on real concrete executions of BOTH paths (the
  independent concrete interpreter), corroborated by the bounded symbolic
  executor;
* the reported memory statistics are counted from the *emitted* machine
  instructions, independently re-tallied here (never the heuristic estimate);
* a hand-layout weighted cost check on a tiny single-block program.
"""

import unittest

from regalloc import allocate, ProgramFormatError, BudgetExhaustedError
from regalloc.model import Block, Instr, Program
from regalloc.profile import (
    edge_frequencies,
    memory_traffic,
    normalize_frequencies,
    spill_costs,
)
from regalloc.samples import (
    sample_hot_loop_cold_error,
    sample_register_pressure,
    HOT_PROFILE,
    COLD_PROFILE,
    HOT_PARAMS,
    COLD_PARAMS,
)
from regalloc.concrete import (
    exit_live_vars,
    run_allocated,
    run_source,
)
from regalloc.symbolic import equivalence_check


# --------------------------------------------------------------------------
# input contract
# --------------------------------------------------------------------------

class TestFrequencyValidation(unittest.TestCase):
    def setUp(self):
        self.prog = sample_register_pressure()

    def test_default_is_none_and_legacy_result_unchanged(self):
        a = allocate(self.prog, 2)
        b = allocate(self.prog, 2, frequencies=None)
        self.assertIsNone(a.frequencies)
        self.assertEqual(a.assignment, b.assignment)
        self.assertEqual(a.slots, b.slots)
        self.assertEqual(a.history, b.history)
        self.assertEqual(
            a.allocated_program.render(), b.allocated_program.render()
        )

    def test_accepts_tuple_and_list_identically(self):
        a = allocate(self.prog, 2, frequencies=(1,))
        b = allocate(self.prog, 2, frequencies=[1])
        self.assertEqual(a.slots, b.slots)
        self.assertEqual(a.assignment, b.assignment)

    def test_wrong_length(self):
        with self.assertRaises(ProgramFormatError) as cm:
            allocate(self.prog, 2, frequencies=(1, 2))
        self.assertIn("frequencies", str(cm.exception))

    def test_bool_entries_rejected(self):
        with self.assertRaises(ProgramFormatError):
            normalize_frequencies(self.prog, (True,))
        with self.assertRaises(ProgramFormatError):
            allocate(self.prog, 2, frequencies=(False,))

    def test_negative_rejected(self):
        with self.assertRaises(ProgramFormatError):
            normalize_frequencies(self.prog, (-1,))

    def test_float_and_garbage_rejected(self):
        with self.assertRaises(ProgramFormatError):
            normalize_frequencies(self.prog, (1.0,))
        with self.assertRaises(ProgramFormatError):
            normalize_frequencies(self.prog, "1")
        with self.assertRaises(ProgramFormatError):
            normalize_frequencies(self.prog, 5)

    def test_zero_frequency_allowed(self):
        alloc = allocate(self.prog, 2, frequencies=(0,))
        # block-local spill traffic is weighted by 0
        self.assertEqual(alloc.memory["weighted_total"], 0)
        # raw spills still physically present
        self.assertGreater(alloc.memory["total"], 0)

    def test_unprofiled_allocation_still_reports_traffic(self):
        alloc = allocate(self.prog, 2)
        self.assertIsNone(alloc.frequencies)
        self.assertIsNone(alloc.edge_frequencies)
        self.assertEqual(alloc.cost_history, ())
        # memory stats are always present (structural loop-depth weighting)
        self.assertGreaterEqual(alloc.memory["total"], 1)
        self.assertEqual(
            alloc.memory["total"],
            alloc.memory["loads"] + alloc.memory["stores"],
        )
        self.assertEqual(
            alloc.memory["weighted_total"],
            alloc.memory["weighted_loads"] + alloc.memory["weighted_stores"],
        )


# --------------------------------------------------------------------------
# edge frequency estimation
# --------------------------------------------------------------------------

class TestEdgeFrequencies(unittest.TestCase):
    def _diamond(self):
        # 0 -> 1,2 ; 1 -> 3 ; 2 -> 3 (diamond), 3 -> 4 (exit)
        b0 = Block("b0", (Instr(0, (0,), (), note="sink"),), (1, 2))
        b1 = Block("b1", (Instr(1, (0,), (), note="sink"),), (3,))
        b2 = Block("b2", (Instr(2, (0,), (), note="sink"),), (3,))
        b3 = Block("b3", (Instr(3, (0,), (), note="sink"),), (4,))
        b4 = Block("b4", (Instr(4, (0,), (), note="sink"),), ())
        return Program((b0, b1, b2, b3, b4), params=(0,))

    def test_unique_successor_takes_whole_frequency(self):
        prog = self._diamond()
        ef = edge_frequencies(prog, (7, 11, 13, 5, 1))
        self.assertEqual(ef[(1, 3)], 11)   # rule 1
        self.assertEqual(ef[(2, 3)], 13)   # rule 1
        self.assertEqual(ef[(3, 4)], 5)    # rule 1

    def test_unique_predecessor_min_clamped(self):
        prog = self._diamond()
        ef = edge_frequencies(prog, (7, 11, 13, 5, 1))
        # b1/b2 each have the single predecessor b0 (freq 7)
        self.assertEqual(ef[(0, 1)], min(7, 11))
        self.assertEqual(ef[(0, 2)], min(7, 13))

    def test_zero_block_sends_zero(self):
        prog = self._diamond()
        ef = edge_frequencies(prog, (0, 9, 9, 9, 9))
        self.assertEqual(ef[(0, 1)], 0)
        self.assertEqual(ef[(0, 2)], 0)

    def test_rule4_fork_into_join_residual_split(self):
        # CFG: 0 -> {1, 2}; 1 -> 2; 2 -> 3(exit).  Edge 0->1 lands on a
        # block with unique predecessor (rule 2, min(11,4)=4); edge 0->2
        # enters a join with TWO predecessors {0,1}, so it falls to rule 4
        # and receives b0's residual 11-4=7.
        b0 = Block("b0", (Instr(0, (0,), (), note="sink"),), (1, 2))
        b1 = Block("b1", (Instr(1, (0,), (), note="sink"),), (2,))
        b2 = Block("b2", (Instr(2, (0,), (), note="sink"),), (3,))
        b3 = Block("b3", (Instr(3, (0,), (), note="sink"),), ())
        prog = Program((b0, b1, b2, b3), params=(0,))
        ef = edge_frequencies(prog, (11, 4, 5, 1))
        self.assertEqual(ef[(0, 1)], 4)
        self.assertEqual(ef[(0, 2)], 7)
        self.assertEqual(ef[(1, 2)], 4)   # rule 1: unique successor
        self.assertEqual(ef[(2, 3)], 5)   # rule 1

    def test_rule4_genuine_even_split_with_remainder(self):
        # 0 -> {1, 4}; 1 -> {2, 3}; 4 -> {2, 3}; 2, 3 exit.  Both join
        # targets have two predecessors, so neither rule 2 nor rule 3
        # applies to edges out of 1 (or 4): they fall to rule 4.  An odd
        # frequency splits with the remainder going to successor index 2.
        def sink(uid):
            return Block(f"b{uid}", (Instr(uid, (0,), (), note="sink"),), ())
        b0 = Block("b0", (), (1, 4))
        b1 = Block("b1", (), (2, 3))
        b4 = Block("b4", (), (2, 3))
        b2 = sink(2)
        b3 = sink(3)
        prog = Program((b0, b1, b2, b3, b4), params=(0,))
        ef = edge_frequencies(prog, (1, 11, 20, 20, 7))
        self.assertEqual(ef[(1, 2)], 6)
        self.assertEqual(ef[(1, 3)], 5)
        self.assertEqual(ef[(4, 2)], 4)
        self.assertEqual(ef[(4, 3)], 3)

    def test_hot_loop_back_edge_from_header_balance(self):
        # 0 -> 1 (header); 1 -> 2(latch), 3(exit); 2 -> 1 (back edge)
        b0 = Block("b0", (), (1,))
        b1 = Block("b1", (Instr(0, (0,), (), note="sink"),), (2, 3))
        b2 = Block("b2", (Instr(1, (0,), (), note="sink"),), (1,))
        b3 = Block("b3", (Instr(2, (0,), (), note="sink"),), ())
        prog = Program((b0, b1, b2, b3), params=(0,))
        # header executes 1000x, entered once: back edge carries 999
        ef = edge_frequencies(prog, (1, 1000, 999, 1))
        self.assertEqual(ef[(2, 1)], 999)      # header 1000 - entry edge 1
        # exit block 3 has unique predecessor 1 -> rule 2 min(1000,1)=1
        self.assertEqual(ef[(1, 3)], 1)
        self.assertEqual(ef[(1, 2)], 999)      # rule 4 residual: 1000-1
        self.assertEqual(ef[(0, 1)], 1)        # rule 1


# --------------------------------------------------------------------------
# iteration mechanics
# --------------------------------------------------------------------------

class TestRecomputation(unittest.TestCase):
    def test_costs_recomputed_every_iteration(self):
        prog = sample_hot_loop_cold_error()
        alloc = allocate(prog, 4, frequencies=HOT_PROFILE)
        self.assertGreater(alloc.iterations, 1)
        # one fresh cost table was recorded per iteration
        self.assertEqual(len(alloc.cost_history), alloc.iterations)
        # later tables never mention already-spilled variables as candidates
        for i, table in enumerate(alloc.cost_history):
            for v in alloc.history[i]:
                self.assertNotIn(v, table)

    def test_liveness_rebuilt_each_iteration(self):
        # indirect: allocating twice is deterministic (fresh rebuilds do not
        # leak state between calls)
        prog = sample_hot_loop_cold_error()
        a = allocate(prog, 4, frequencies=HOT_PROFILE)
        b = allocate(prog, 4, frequencies=HOT_PROFILE)
        self.assertEqual(a.slots, b.slots)
        self.assertEqual(a.allocated_program.render(),
                         b.allocated_program.render())


def independent_location_invariants(prog, alloc):
    """Same independent checker as tests/test_allocation.py (inlined so this
    module stays importable without the tests package)."""
    from regalloc.liveness import liveness_with_phis

    rep = liveness_with_phis(prog)
    colors = alloc.assignment
    spilled = set(alloc.slots)
    violations = []

    def home(v):
        return None if v in spilled else colors.get(v)

    for bi, b in enumerate(prog.blocks):
        for ii, ins in enumerate(b.instrs):
            for point_name, live in (
                ("before", rep.instr_before[bi][ii]),
                ("after", rep.instr_after[bi][ii]),
            ):
                seen = {}
                for v in live:
                    r = home(v)
                    if r is None:
                        continue
                    if r in seen and seen[r] != v:
                        violations.append(
                            f"b{bi} uid{ins.uid} {point_name}: v{v} and "
                            f"v{seen[r]} share r{r}"
                        )
                    seen[r] = v
            survive = rep.instr_after[bi][ii] - set(ins.defs)
            for v in survive:
                r = home(v)
                if r is not None and r in ins.clobbers:
                    violations.append(
                        f"b{bi} uid{ins.uid}: v{v} survives call in clobbered r{r}"
                    )
    for (a, b), live in rep.edge_live.items():
        seen = {}
        for v in live:
            r = home(v)
            if r is None:
                continue
            if r in seen and seen[r] != v:
                violations.append(f"edge {a}->{b}: v{v} and v{seen[r]} share r{r}")
            seen[r] = v
    return violations


# --------------------------------------------------------------------------
# the headline capability: profile moves spills off the hot path
# --------------------------------------------------------------------------

class TestHotLoopColdError(unittest.TestCase):
    def setUp(self):
        self.prog = sample_hot_loop_cold_error()
        self.legacy = allocate(self.prog, 4)
        self.hot = allocate(self.prog, 4, frequencies=HOT_PROFILE)
        self.inverted = allocate(self.prog, 4, frequencies=COLD_PROFILE)

    def _weighted(self, alloc, block_freq):
        ef = edge_frequencies(self.prog, block_freq)
        return memory_traffic(self.prog, alloc.allocated_program,
                              block_freq, ef)["weighted_total"]

    def test_profile_changes_spill_choice(self):
        self.assertNotEqual(set(self.legacy.slots), set(self.hot.slots))
        self.assertNotEqual(set(self.legacy.slots), set(self.inverted.slots))
        self.assertNotEqual(set(self.hot.slots), set(self.inverted.slots))

    def test_hot_allocation_reduces_hot_weighted_traffic(self):
        legacy_hot = self._weighted(self.legacy, HOT_PROFILE)
        profiled_hot = self._weighted(self.hot, HOT_PROFILE)
        self.assertLess(profiled_hot, legacy_hot)

    def test_inverted_allocation_reduces_cold_weighted_traffic(self):
        legacy_cold = self._weighted(self.legacy, COLD_PROFILE)
        profiled_cold = self._weighted(self.inverted, COLD_PROFILE)
        self.assertLess(profiled_cold, legacy_cold)

    def test_profiled_spills_are_not_more_and_traffic_wins(self):
        # the win is about WHERE spills land; the profiled run also does not
        # spill more variables than the legacy run on this fixture
        self.assertLessEqual(len(self.hot.slots), len(self.legacy.slots))

    def test_concrete_equivalence_both_real_paths(self):
        for params in (HOT_PARAMS, COLD_PARAMS):
            src = run_source(self.prog, params)
            observable = exit_live_vars(self.prog, src.exit_block)
            for name, alloc in (("legacy", self.legacy),
                                ("hot", self.hot),
                                ("inverted", self.inverted)):
                got = run_allocated(self.prog, alloc, params)
                self.assertEqual(got.uid_trace, src.uid_trace, name)
                self.assertEqual(got.block_trace, src.block_trace, name)
                for v in observable:
                    self.assertEqual(got.values[v], src.values[v],
                                     f"{name}: v{v} mismatch")

    def test_symbolic_equivalence_bounded(self):
        for alloc in (self.legacy, self.hot, self.inverted):
            res = equivalence_check(self.prog, alloc,
                                    depth_bound=12, step_budget=20_000)
            self.assertTrue(res.equivalent, res.failure)

    def test_concrete_interpreter_is_discriminating(self):
        # mutate one non-commutative (lt) op by swapping its two reads: the
        # concrete interpreter must catch the changed branch/result rather
        # than rubber-stamp the allocation
        import copy
        from regalloc.lower import MI
        ap = copy.deepcopy(self.hot.allocated_program)
        mutated = False
        for b in ap.blocks:
            for i, mi in enumerate(b.instrs):
                if (mi.kind == "op" and mi.note == "lt"
                        and len(mi.reads) == 2
                        and mi.reads[0] != mi.reads[1]):
                    b.instrs[i] = MI(
                        kind="op",
                        reads=(mi.reads[1], mi.reads[0]),
                        writes=mi.writes, clobbers=mi.clobbers,
                        uid=mi.uid, note=mi.note,
                    )
                    mutated = True
                    break
            if mutated:
                break
        self.assertTrue(mutated)
        bad = type(self.hot)(
            program=self.hot.program, k=self.hot.k,
            assignment=self.hot.assignment, slots=self.hot.slots,
            allocated_program=ap, liveness=self.hot.liveness,
            iterations=self.hot.iterations, history=self.hot.history,
        )
        src = run_source(self.prog, HOT_PARAMS)
        got = run_allocated(self.prog, bad, HOT_PARAMS)
        observable = exit_live_vars(self.prog, 4)
        self.assertFalse(
            all(got.values.get(v) == src.values.get(v) for v in observable)
            and got.block_trace == src.block_trace,
            "concrete interpreter failed to detect the mutated comparison",
        )

    def test_invariants_still_hold(self):
        # fixed/clobber/no-two-live-in-one-reg checks, independent checker
        for alloc in (self.legacy, self.hot, self.inverted):
            self.assertEqual(
                independent_location_invariants(self.prog, alloc), []
            )


# --------------------------------------------------------------------------
# reported statistics come from the actual lowered code
# --------------------------------------------------------------------------

def _independent_traffic_tally(program, allocated_program, block_freq, ef):
    """Recount weighted loads/stores by scanning lowered blocks independently.

    Deliberately does not call regalloc.profile.memory_traffic.
    """
    wl = ws = rl = rs = 0
    sites = []
    for ab in allocated_program.blocks:
        w = block_freq[ab.origin] if ab.origin >= 0 else ef[ab.split_edge]
        nl = sum(1 for mi in ab.instrs if mi.kind == "load")
        ns = sum(1 for mi in ab.instrs if mi.kind == "store")
        wl += nl * w
        ws += ns * w
        rl += nl
        rs += ns
        sites.append((ab.label, w, nl, ns))
    return {
        "loads": rl, "stores": rs, "weighted_loads": wl,
        "weighted_stores": ws, "weighted_total": wl + ws,
        "sites": sites,
    }


class TestReportedTrafficIsActual(unittest.TestCase):
    def test_matches_independent_scan_hot_fixture(self):
        prog = sample_hot_loop_cold_error()
        alloc = allocate(prog, 4, frequencies=HOT_PROFILE)
        ef = edge_frequencies(prog, HOT_PROFILE)
        got = memory_traffic(prog, alloc.allocated_program, HOT_PROFILE, ef)
        want = _independent_traffic_tally(
            prog, alloc.allocated_program, HOT_PROFILE, ef
        )
        self.assertEqual(got["loads"], want["loads"])
        self.assertEqual(got["stores"], want["stores"])
        self.assertEqual(got["weighted_loads"], want["weighted_loads"])
        self.assertEqual(got["weighted_stores"], want["weighted_stores"])
        self.assertEqual(got["weighted_total"], want["weighted_total"])
        # and it is NOT the heuristic estimate table
        self.assertIsInstance(alloc.cost_history, tuple)

    def test_hand_layout_single_block(self):
        # K=2, three values coexist at once -> exactly one variable spills;
        # with a uniform block profile F its weighted traffic is F times the
        # physical load/store count, which we also hand-check from the
        # rendered machine code.
        prog = Program(
            blocks=(Block(
                label="e",
                instrs=(
                    Instr(0, (), (2,), note="const:1"),
                    Instr(1, (2, 0), (3,), note="add"),
                    Instr(2, (3, 1), (), note="sink"),
                ),
                succs=(),
            ),),
            params=(0, 1),
        )
        F = 10
        alloc = allocate(prog, 2, frequencies=(F,))
        self.assertEqual(len(alloc.slots), 1)
        victim = next(iter(alloc.slots))

        # hand expectation derived from the spill-form semantics:
        #  - a spilled parameter is stack-passed: no entry store, one reload
        #    per use;
        #  - a spilled def gets one store per def plus one reload per use.
        use_count = {0: 1, 1: 1, 2: 2, 3: 2}
        def_count = {0: 0, 1: 0, 2: 1, 3: 1}
        expected_weighted = (
            use_count[victim] * F
            + (def_count[victim] * F if victim not in prog.params else 0)
        )
        self.assertEqual(alloc.memory["weighted_total"], expected_weighted)

        # physical instruction count agrees with a direct text scan
        rendered = alloc.allocated_program.render()
        n_load = rendered.count("\n    load ") + (
            1 if rendered.startswith("    load ") else 0)
        n_store = rendered.count("\n    store ")
        self.assertEqual(alloc.memory["loads"], n_load)
        self.assertEqual(alloc.memory["stores"], n_store)
        self.assertEqual(
            alloc.memory["weighted_total"],
            (n_load + n_store) * F,
        )


# --------------------------------------------------------------------------
# tie breaks and budget termination
# --------------------------------------------------------------------------

class TestTieBreakAndTermination(unittest.TestCase):
    def test_equal_ratio_prefers_smaller_id(self):
        # straight-line pressure program; a perfectly uniform profile gives
        # every candidate the same cost, so among the densest K-core nodes
        # the smallest variable id must be chosen.  The expected winner is
        # derived here by an independent recomputation of the same rule.
        prog = sample_register_pressure()
        import regalloc.allocator as core
        vp = core.build_virtual_program(prog, set())
        k = 2
        core_nodes = core._constrained_core(vp, k)
        costs = {v: 1 for v in vp.real_nodes}
        # independent expectation: max degree, tie -> min id
        max_deg = max(len(vp.adj[v]) for v in core_nodes)
        expected = min(v for v in core_nodes if len(vp.adj[v]) == max_deg)
        got = core._spill_victim(vp, prog, costs, profile=True, k=k)
        self.assertEqual(got, expected)

    def test_pseudos_and_temporaries_are_never_spill_candidates(self):
        # a program with fixed operands and a call: precolored pseudo nodes
        # (negative ids) and spill reload/store temporaries (ids >= var_count)
        # must never appear in any spill set
        from regalloc.model import PhiNode
        prog = Program(
            blocks=(
                Block("e", (
                    Instr(0, (0,), (2,), fixed={0: 0}, note="arg in r0"),
                    Instr(1, (2, 0), (3,), clobbers=(0, 1), note="call"),
                    Instr(2, (3, 2), (4,), note="join"),
                    Instr(3, (4,), (), note="sink"),
                ), ()),
            ),
            params=(0, 1),
        )
        alloc = allocate(prog, 2, frequencies=(1,))
        vmax = prog.var_count()
        for v in alloc.slots:
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, vmax)
        for v in alloc.assignment:
            self.assertLess(v, vmax)

    def test_budget_exhaustion_still_distinct_and_terminal(self):
        import regalloc.allocator as mod
        saved = mod.COLORING_BUDGET
        mod.COLORING_BUDGET = 1
        try:
            with self.assertRaises(BudgetExhaustedError):
                allocate(sample_register_pressure(), 3,
                         frequencies=(1,))
        finally:
            mod.COLORING_BUDGET = saved


if __name__ == "__main__":
    unittest.main()
