"""Tests for the block-frequency-aware spill heuristic.

Coverage:

* strict validation of the new ``frequencies`` input (length, sign, bool,
  wrong type, dict construction);
* frequencies are heuristic-only: programs that need no spilling get the
  *identical* allocation with/without/with-inverted hints, and every
  frequency-driven result passes the paired symbolic executor;
* the spill decision log (``spill_decisions``) records the mode and the
  victim, with variable-id tie breaks;
* the heuristic cost table is hand-checked on a tiny program (every value
  derived on paper, not by calling the allocator);
* the *actual* traffic accounting (``access_stats``) is counted independently
  straight from the lowered machine instructions, including the phi scratch
  slot used to break a register cycle;
* the hot-loop/cold-recovery fixture really does move weighted memory traffic
  out of the hot blocks compared with the original policy, and the inverted
  profile reverses the decision;
* fixed registers, call clobbers and the "no spilling short reload/store
  temporaries" rules keep holding;
* budget exhaustion in frequency mode is still reported distinctly.
"""

import os
import sys
import unittest

from regalloc import (
    BudgetExhaustedError,
    ProgramFormatError,
    actual_access_stats,
    allocate,
    spill_cost_table,
)
from regalloc.cost import VarSpillCost
from regalloc.model import Block, Instr, PhiNode, Program, program_from_dict
from regalloc.samples import (
    sample_call_surviving,
    sample_hot_cold_loop,
    sample_register_pressure,
)
from regalloc.symbolic import equivalence_check

# discover -s tests puts the tests directory on sys.path; a named module run
# does not.  Reuse the independent invariant checker from the sibling test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_allocation import independent_location_invariants  # noqa: E402


# --------------------------------------------------------------------------
# strict input validation
# --------------------------------------------------------------------------

class TestFrequencyValidation(unittest.TestCase):
    def _prog(self, frequencies):
        return Program(
            blocks=(
                Block(label="e",
                      instrs=(Instr(uid=0, uses=(0,), defs=(1,)),),
                      succs=()),
            ),
            params=(0,),
            frequencies=frequencies,
        )

    def test_none_tuple_list_zero_accepted(self):
        self.assertIsNone(self._prog(None).frequencies)
        self.assertEqual(self._prog((1,)).frequencies, (1,))
        self.assertEqual(self._prog((0,)).frequencies, (0,))
        # lists are accepted and normalized to a tuple of ints
        prog = Program(
            blocks=(
                Block(label="a", instrs=(), succs=(1,)),
                Block(label="b", instrs=(), succs=()),
            ),
            frequencies=[0, 0],
        )
        self.assertEqual(prog.frequencies, (0, 0))

    def test_wrong_length_rejected(self):
        with self.assertRaises(ProgramFormatError) as cm:
            self._prog((1, 2))  # one block, two frequencies
        self.assertIn("frequencies", str(cm.exception))
        with self.assertRaises(ProgramFormatError) as cm:
            self._prog(())
        self.assertIn("frequencies", str(cm.exception))

    def test_negative_rejected(self):
        with self.assertRaises(ProgramFormatError):
            self._prog((-1,))

    def test_bool_rejected(self):
        with self.assertRaises(ProgramFormatError):
            self._prog((True,))

    def test_float_rejected(self):
        with self.assertRaises(ProgramFormatError):
            self._prog((1.0,))

    def test_wrong_container_rejected(self):
        with self.assertRaises(ProgramFormatError):
            self._prog({0: 1})
        with self.assertRaises(ProgramFormatError):
            self._prog("1")

    def test_from_dict_supports_frequencies(self):
        data = {
            "blocks": [{
                "label": "e",
                "instrs": [{"uid": 0, "uses": [0], "defs": [1]}],
                "succs": [],
            }],
            "params": [0],
            "frequencies": [3],
        }
        prog = program_from_dict(data)
        self.assertEqual(prog.frequencies, (3,))
        bad = dict(data)
        bad["frequencies"] = [3, 4]
        with self.assertRaises(ProgramFormatError):
            program_from_dict(bad)

    def test_omitted_in_dict_means_none(self):
        prog = program_from_dict({
            "blocks": [{"instrs": [{"uid": 0, "uses": [0], "defs": [1]}]}],
            "params": [0],
        })
        self.assertIsNone(prog.frequencies)


# --------------------------------------------------------------------------
# frequencies influence heuristics only, never semantics
# --------------------------------------------------------------------------

class TestFrequencyIsHeuristicOnly(unittest.TestCase):
    def test_no_spill_program_identical_assignment(self):
        # When coloring succeeds without spilling, the heuristic is never
        # consulted, so every profile must produce the exact same result.
        base = sample_call_surviving()
        ref = allocate(base, 4)
        for freqs in ((1,), (0,), (10 ** 6,), (7,)):
            prog = Program(blocks=base.blocks, params=base.params,
                           frequencies=freqs)
            alloc = allocate(prog, 4)
            self.assertEqual(alloc.assignment, ref.assignment)
            self.assertEqual(alloc.slots, ref.slots)
            self.assertEqual(
                alloc.allocated_program.render(),
                ref.allocated_program.render(),
            )

    def test_every_hot_cold_result_is_equivalent(self):
        for inverted in (False, True):
            prog = sample_hot_cold_loop(inverted=inverted)
            alloc = allocate(prog, 3)
            res = equivalence_check(prog, alloc, depth_bound=24, step_budget=30000)
            self.assertTrue(res.equivalent, msg=res.failure)
            self.assertEqual(independent_location_invariants(prog, alloc), [])

    def test_default_mode_uses_original_policy(self):
        prog = sample_register_pressure()
        self.assertIsNone(prog.frequencies)
        alloc = allocate(prog, 2)
        self.assertIsNone(alloc.frequencies)
        self.assertTrue(all(d.mode == "loop-depth" for d in alloc.spill_decisions))
        self.assertTrue(all(d.weighted_cost is None for d in alloc.spill_decisions))
        # decisions match the recorded spill history one victim per round
        victims = tuple(d.victim for d in alloc.spill_decisions)
        spilled_so_far: list[int] = []
        for i, v in enumerate(victims):
            self.assertEqual(alloc.history[i], tuple(sorted(spilled_so_far)))
            spilled_so_far.append(v)

    def test_frequency_mode_records_mode_and_rebuild_rounds(self):
        alloc = allocate(sample_hot_cold_loop(), 3)
        self.assertTrue(all(d.mode == "frequency" for d in alloc.spill_decisions))
        self.assertTrue(all(d.weighted_cost is not None for d in alloc.spill_decisions))
        # history strictly grows by exactly the chosen victim each round
        for d in alloc.spill_decisions:
            self.assertEqual(
                alloc.history[d.iteration],
                tuple(sorted(set(x.victim for x in alloc.spill_decisions
                                 if x.iteration <= d.iteration))),
            )

    def test_default_policy_choice_sequence_is_unchanged_from_baseline(self):
        # Locks the "no frequencies -> original strategy" promise: the exact
        # spill-victim order on the baseline fixture must not move when the
        # frequency feature is present.
        alloc = allocate(sample_register_pressure(), 2)
        self.assertEqual(
            [d.victim for d in alloc.spill_decisions],
            [0, 3, 4, 1, 2, 5],
        )


# --------------------------------------------------------------------------
# hand-checked heuristic cost table
# --------------------------------------------------------------------------

def _hand_cost_program(frequencies):
    """
    b0 entry (f0): uid0 v1 = f()
                  uid1 v2 = f(v1)
                  -> b1
    b1 header (f1): phi v3 <- v2 (from b0) | v5 (from b2)
                    uid2 v4 = f(v3)
                    -> b2 | b3
    b2 (f2): uid3 v5 = f(v4) -> b1
    b3 exit (f3): uid4 sink(v4)
    """
    b0 = Block(label="b0", instrs=(
        Instr(uid=0, uses=(), defs=(1,)),
        Instr(uid=1, uses=(1,), defs=(2,)),
    ), succs=(1,))
    b1 = Block(label="b1", instrs=(
        Instr(uid=2, uses=(3,), defs=(4,)),
    ), succs=(2, 3),
        phis=(PhiNode(def_var=3, sources=((0, 2), (2, 5))),))
    b2 = Block(label="b2", instrs=(
        Instr(uid=3, uses=(4,), defs=(5,)),
    ), succs=(1,))
    b3 = Block(label="b3", instrs=(
        Instr(uid=4, uses=(4,), defs=()),
    ), succs=())
    return Program(blocks=(b0, b1, b2, b3), frequencies=frequencies)


class TestHandCheckedCostTable(unittest.TestCase):
    def test_paper_values(self):
        f0, f1, f2, f3 = 7, 11, 13, 17
        prog = _hand_cost_program((f0, f1, f2, f3))
        table = spill_cost_table(prog, prog.frequencies)

        def check(v, loads, stores, wl, ws):
            c = table[v]
            self.assertEqual((c.loads, c.stores, c.weighted_loads, c.weighted_stores),
                             (loads, stores, wl, ws), msg=f"v{v}")

        # v1: def + one body use, both in b0
        check(1, 1, 1, f0, f0)
        # v2: def in b0 (store), phi source on edge 0->1 (reload @ f0)
        check(2, 1, 1, f0, f0)
        # v3: phi def stored on both incoming edges (f0 and f2), one body use
        check(3, 1, 2, f1, f0 + f2)
        # v4: def in b1; used in b2 and b3
        check(4, 2, 1, f2 + f3, f1)
        # v5: def in b2; phi source on edge 2->1 (reload @ f2)
        check(5, 1, 1, f2, f2)

    def test_zero_frequency_cold_branch_is_free_weighted(self):
        prog = _hand_cost_program((1, 1, 0, 1))  # b2 never runs
        table = spill_cost_table(prog, prog.frequencies)
        # v4's use inside the cold b2 contributes static load but no weight;
        # its exit-block use still does
        self.assertEqual(table[4].loads, 2)
        self.assertEqual(table[4].weighted_loads, 1)


# --------------------------------------------------------------------------
# actual traffic accounting, counted independently from lowered code
# --------------------------------------------------------------------------

class TestActualAccessStats(unittest.TestCase):
    def test_stats_present_even_without_frequencies(self):
        alloc = allocate(sample_register_pressure(), 2)
        stats = alloc.access_stats
        self.assertIsNotNone(stats)
        # no hints -> weighted counts equal raw counts (implicit weight 1)
        self.assertEqual(stats.weighted_loads, stats.loads)
        self.assertEqual(stats.weighted_stores, stats.stores)
        # every recorded load/store really exists in the lowered program
        mi_loads = sum(
            1 for b in alloc.allocated_program.blocks for mi in b.instrs
            if mi.kind == "load"
        )
        mi_stores = sum(
            1 for b in alloc.allocated_program.blocks for mi in b.instrs
            if mi.kind == "store"
        )
        self.assertEqual(stats.loads, mi_loads)
        self.assertEqual(stats.stores, mi_stores)

    def test_straight_line_slot_weighted_by_block_frequency(self):
        # three values coexist at the peak, so K=2 forces exactly one spill;
        # the spilled value has one def (store) and one use (reload)
        prog = Program(
            blocks=(Block(label="e", instrs=(
                Instr(uid=0, uses=(), defs=(1,)),
                Instr(uid=1, uses=(), defs=(2,)),
                Instr(uid=2, uses=(), defs=(3,)),
                Instr(uid=3, uses=(1, 2), defs=(4,)),
                Instr(uid=4, uses=(3, 4), defs=(5,)),
                Instr(uid=5, uses=(5,), defs=()),
            ), succs=()),),
            frequencies=(5,),
        )
        alloc = allocate(prog, 2)
        self.assertGreaterEqual(len(alloc.slots), 1)
        stats = alloc.access_stats
        # independent recount straight from the MI stream, per slot
        per: dict[int, list[int]] = {}
        for b in alloc.allocated_program.blocks:
            for mi in b.instrs:
                if mi.kind in ("load", "store"):
                    row = per.setdefault(mi.slot, [0, 0])
                    row[0 if mi.kind == "load" else 1] += 1
        raw_l = sum(r[0] for r in per.values())
        raw_s = sum(r[1] for r in per.values())
        self.assertEqual((raw_l, raw_s), (stats.loads, stats.stores))
        self.assertEqual(stats.weighted_total, 5 * stats.total)
        # every per-slot weighted entry is exactly 5x its static entry, since
        # the single block executes with frequency 5
        for slot, l, s, wl, ws in stats.by_slot:
            self.assertEqual((wl, ws), (5 * l, 5 * s))
            self.assertEqual((l, s), tuple(per[slot]))

    def test_phi_cycle_scratch_slot_is_counted(self):
        # The accounting function is independent of the allocator core: feed
        # it a hand-laid-out AllocatedProgram containing a critical-edge
        # split block that breaks a register permutation cycle through the
        # scratch slot (one store + one load).  The split block inherits its
        # predecessor edge's frequency; origin blocks keep their own.
        from regalloc.lower import MI, ABlock, AllocatedProgram

        scratch = 99
        origin = Block(label="b0", instrs=(), succs=(1,))
        target = Block(label="b1", instrs=(), succs=())
        prog = Program(blocks=(origin, target), frequencies=(3, 9))

        ab0 = ABlock(label="b0", succs=[2], origin=0)
        ab0.instrs.append(MI(kind="load", dst=0, slot=5))    # normal reload @3
        ab0.instrs.append(MI(kind="store", src=1, slot=5))   # normal store  @3
        split = ABlock(label="b0__split_b1", succs=[1], origin=-1,
                       split_edge=(0, 1))
        split.instrs.append(MI(kind="store", src=0, slot=scratch))  # cycle @3
        split.instrs.append(MI(kind="load", dst=1, slot=scratch))   # cycle @3
        ab1 = ABlock(label="b1", succs=[], origin=1)
        ap = AllocatedProgram(
            blocks=(ab0, split, ab1), k=2, slots={5: 5},
            assignment={}, scratch_slot=scratch, param_regs={},
        )

        class _Holder:
            pass

        holder = _Holder()
        holder.program = prog
        holder.allocated_program = ap
        stats = actual_access_stats(prog, holder)

        self.assertEqual(stats.loads, 2)
        self.assertEqual(stats.stores, 2)
        # all four live on the edge out of b0, whose frequency is 3
        self.assertEqual(stats.weighted_loads, 6)
        self.assertEqual(stats.weighted_stores, 6)
        self.assertEqual(
            dict((row[0], row) for row in stats.by_slot),
            {
                5: (5, 1, 1, 3, 3),
                scratch: (scratch, 1, 1, 3, 3),
            },
        )
        self.assertEqual(stats.weighted_total,
                         sum(wl + ws for _, _, _, wl, ws in stats.by_slot))

    def test_split_block_inherits_predecessor_frequency(self):
        from regalloc.lower import MI, ABlock, AllocatedProgram

        scratch = 7
        origin = Block(label="b0", instrs=(), succs=(1,))
        target = Block(label="b1", instrs=(), succs=())
        prog = Program(blocks=(origin, target), frequencies=(4, 100))
        ab0 = ABlock(label="b0", succs=[2], origin=0)
        split = ABlock(label="s", succs=[1], origin=-1, split_edge=(0, 1))
        # one spill reload on the edge: must take b0's weight 4, not b1's 100
        split.instrs.append(MI(kind="load", dst=0, slot=3))
        ab1 = ABlock(label="b1", succs=[], origin=1)
        ap = AllocatedProgram(
            blocks=(ab0, split, ab1), k=2, slots={3: 3},
            assignment={}, scratch_slot=scratch, param_regs={},
        )

        class _Holder:
            pass

        holder = _Holder()
        holder.allocated_program = ap
        stats = actual_access_stats(prog, holder)
        self.assertEqual(stats.loads, 1)
        self.assertEqual(stats.weighted_loads, 4)


# --------------------------------------------------------------------------
# the demonstrated win: hot loop vs cold error recovery
# --------------------------------------------------------------------------

class TestHotColdHeuristic(unittest.TestCase):
    def test_frequency_policy_moves_traffic_out_of_hot_blocks(self):
        hot = sample_hot_cold_loop()
        # same program shape with no hints -> original loop-depth policy
        old_shape = Program(blocks=hot.blocks, params=hot.params, frequencies=None)
        old_alloc = allocate(old_shape, 3)
        new_alloc = allocate(hot, 3)

        # original heuristic is fooled by the cold block's many static uses
        # (they sit inside the natural loop and get the x10 depth weight):
        # it spills hot invariants v1 and v3.
        self.assertIn(1, old_alloc.slots)
        self.assertIn(3, old_alloc.slots)
        self.assertNotIn(2, old_alloc.slots)

        # frequency policy spills the cold value v2 instead of v3, and its
        # first-round victims in order are v2 (weight 1) then v1 (101)
        victims = [d.victim for d in new_alloc.spill_decisions]
        self.assertEqual(victims[0], 2)
        self.assertNotIn(3, new_alloc.slots)
        self.assertIn(2, new_alloc.slots)

        # evaluate the OLD lowering against the hot profile (apples to
        # apples) and compare with the traffic actually emitted by the new
        # policy: weighted hot memory accesses must strictly decrease
        old_under_hot = actual_access_stats(hot, old_alloc)
        self.assertGreater(
            old_under_hot.weighted_total,
            new_alloc.access_stats.weighted_total,
        )
        # static traffic is allowed to grow (cold block): 12 cold reloads
        self.assertGreater(new_alloc.access_stats.loads, old_under_hot.loads)
        # new policy: the only hot reload is v1 once per loop visit (100);
        # old policy: v1 and v3 each reloaded on the hot path (200), plus a
        # one-off entry-block reload of the spilled parameter (weight 1).
        self.assertEqual(new_alloc.access_stats.weighted_loads, 100)
        self.assertEqual(old_under_hot.weighted_loads, 202)

    def test_inverted_profile_reverses_the_decision(self):
        inv = sample_hot_cold_loop(inverted=True)
        alloc = allocate(inv, 3)
        # cold-recovery block is now the hot one: v2 must be kept, v3 spilled
        self.assertNotIn(2, alloc.slots)
        self.assertIn(3, alloc.slots)
        # weighted traffic stays in the (now cold) old loop body: only 1
        self.assertLessEqual(alloc.access_stats.weighted_total, 3)

    def test_fixed_and_clobber_constraints_still_hold(self):
        for inverted in (False, True):
            prog = sample_hot_cold_loop(inverted=inverted)
            alloc = allocate(prog, 3)
            self.assertEqual(independent_location_invariants(prog, alloc), [])

    def test_spilled_set_only_contains_real_variables(self):
        alloc = allocate(sample_hot_cold_loop(), 3)
        real = set()
        for b in alloc.program.blocks:
            for ins in b.instrs:
                real.update(ins.defs)
            for phi in b.phis:
                real.add(phi.def_var)
        real.update(alloc.program.params)
        self.assertTrue(set(alloc.slots) <= real)
        for d in alloc.spill_decisions:
            self.assertIn(d.victim, real)
            self.assertGreaterEqual(d.victim, 0)


class TestTieBreakById(unittest.TestCase):
    """Direct checks of the lexicographic rule:
    (degree+1) desc, then weighted cost asc, then variable id asc."""

    def _vp(self, adj):
        from regalloc.allocator import VirtualProgram

        nodes = set(adj)
        return VirtualProgram(
            adj={v: set(ns) for v, ns in adj.items()},
            precolor={},
            instr_plans={},
            edge_plans={},
            real_nodes=set(nodes),
            temp_nodes=set(),
            spilled=set(),
            loop_depth={},
        )

    def test_smaller_id_wins_when_degree_and_cost_tie(self):
        from regalloc.allocator import _frequency_spill_victim

        vp = self._vp({
            3: {7}, 7: {3},
            1: {2}, 2: {1},
        })
        same = VarSpillCost(weighted_loads=5)
        costs = {1: same, 2: same, 3: same, 7: same}
        # all degree 1, all cost 5 -> smallest id
        self.assertEqual(_frequency_spill_victim(vp, costs), 1)

    def test_cost_decides_among_equal_degree(self):
        from regalloc.allocator import _frequency_spill_victim

        # every node has degree 2, so the first key ties and cost decides
        vp = self._vp({
            1: {8, 9},
            2: {8, 9},
            8: {1, 2},
            9: {1, 2},
        })
        costs = {
            1: VarSpillCost(weighted_loads=9, weighted_stores=1),   # hot
            2: VarSpillCost(weighted_loads=0, weighted_stores=1),   # cold
            8: VarSpillCost(weighted_loads=50),
            9: VarSpillCost(weighted_loads=50),
        }
        self.assertEqual(_frequency_spill_victim(vp, costs), 2)

    def test_conflict_relief_outranks_cost(self):
        from regalloc.allocator import _frequency_spill_victim

        vp = self._vp({
            1: {2, 3, 4, 5},
            2: {1}, 3: {1}, 4: {1}, 5: {1},
            8: {9}, 9: {8},
        })
        costs = {
            1: VarSpillCost(weighted_loads=1000),  # very hot but high degree
            2: VarSpillCost(), 3: VarSpillCost(),
            4: VarSpillCost(), 5: VarSpillCost(),
            8: VarSpillCost(), 9: VarSpillCost(),
        }
        # high-degree v1 relieves the clique despite its huge cost; degree is
        # the first key ("same conflict conditions" gate)
        self.assertEqual(_frequency_spill_victim(vp, costs), 1)


class TestEstimatorMatchesEmittedTraffic(unittest.TestCase):
    """For the finally spilled set, the heuristic's *static* per-variable
    event prediction must equal the load/store instructions actually emitted
    for those slots (the only thing it does not predict is the scratch-slot
    pair used to break a register cycle).  This keeps the honest distinction
    between estimate and result while proving the estimate is well calibrated
    on the real spill form, across every sample and every K."""

    def test_static_events_match_on_all_samples(self):
        from regalloc.model import Program
        from regalloc.samples import (
            sample_call_surviving,
            sample_critical_phi,
            sample_loop_call,
            generate_pressure_program,
        )

        progs = [
            sample_call_surviving(),
            sample_critical_phi(),
            sample_loop_call(),
            generate_pressure_program(width=5),
            generate_pressure_program(seed=7, width=8),
            generate_pressure_program(seed=3, width=6, clobber=False),
            sample_hot_cold_loop(),
            sample_hot_cold_loop(inverted=True),
        ]
        for prog in progs:
            unit = tuple([1] * len(prog.blocks))
            for freqs in (unit, prog.frequencies):
                if freqs is None:
                    continue
                base = Program(blocks=prog.blocks, params=prog.params,
                               frequencies=freqs)
                for k in range(2, 9):
                    alloc = allocate(base, k)
                    if not alloc.slots:
                        continue
                    table = spill_cost_table(base, freqs)
                    pred_loads = sum(table.get(v).loads for v in alloc.slots)
                    pred_stores = sum(table.get(v).stores for v in alloc.slots)
                    scratch = alloc.allocated_program.scratch_slot
                    act_loads = act_stores = 0
                    for slot, l, s, _wl, _ws in alloc.access_stats.by_slot:
                        if slot != scratch:
                            act_loads += l
                            act_stores += s
                    self.assertEqual(
                        (pred_loads, pred_stores),
                        (act_loads, act_stores),
                        msg=f"{base.name} K={k} freqs={freqs}",
                    )


class TestBudgetStillDistinctInFrequencyMode(unittest.TestCase):
    def test_budget_exhausted_raised(self):
        import regalloc.allocator as mod

        prog = sample_hot_cold_loop()
        saved = mod.COLORING_BUDGET
        mod.COLORING_BUDGET = 1
        try:
            with self.assertRaises(BudgetExhaustedError):
                allocate(prog, 3)
        finally:
            mod.COLORING_BUDGET = saved
