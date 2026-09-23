"""Exhaustive small-domain differential verification.

For hundreds of deterministically generated tiny programs (straight-line
SSA-style CFGs with mixed w/x classes, calls and early-clobber defs) on a
4-view machine:

1. an independent brute-force enumerator checks EVERY complete
   var-to-view assignment (at most a few dozen per program) — it never
   imports the solver, its liveness or its lowering;
2. the solver's EXACT view-colorer at spilled-set empty is compared
   directly with the brute answer (found <-> feasible, proven infeasible
   <-> no assignment exists) — the small reference domain is checked
   COMPLETELY, not sampled;
3. when the whole pipeline allocates (possibly spilling), the result must
   pass the independent verifier and the paired bit-level symbolic
   executor.

Expected values therefore come solely from the independent enumerator;
the solver is never used to compute them.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from regalloc import allocate2, verify_allocation
from regalloc.allocator2 import build_virtual_program2, exact_color2
from regalloc.liveness import liveness_v2
from regalloc.machine import Machine, RegView
from regalloc.model import Block, Instr, Program
from regalloc.symbolic2 import equivalence_check2

from _reference2 import brute_feasibility


def _machine():
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
    )


def _generate(seed: int):
    """Random dominance-valid straight-line program with 2..4 vars."""
    rng = random.Random(seed)
    nvars = rng.randint(2, 4)
    nparams = rng.randint(0, min(2, nvars - 1))
    classes = {v: rng.choice(("w", "x")) for v in range(nvars)}
    defined = list(range(nparams))
    remaining = list(range(nparams, nvars))
    instrs = []
    uid = 0
    target_len = rng.randint(4, 7)
    while len(instrs) < target_len:
        uses_pool = list(defined)
        rng.shuffle(uses_pool)
        uses = tuple(uses_pool[: rng.randint(0, min(2, len(uses_pool)))])
        defs = ()
        early = ()
        if remaining and rng.random() < 0.7:
            d = remaining.pop(0)
            defs = (d,)
            defined.append(d)
            if rng.random() < 0.3:
                early = (d,)
        is_call = bool(uses) and rng.random() < 0.25
        instrs.append(
            Instr(uid=uid, uses=uses, defs=defs, early_clobber=early,
                  is_call=is_call)
        )
        uid += 1
        if not remaining and not uses_pool and not defs:
            break
    # ensure every remaining var gets defined
    while remaining:
        d = remaining.pop(0)
        instrs.append(Instr(uid=uid, uses=(), defs=(d,)))
        uid += 1
    b = Block(label="e", instrs=tuple(instrs), succs=())
    return Program(
        blocks=(b,),
        params=tuple(range(nparams)),
        name=f"rand-{seed}",
        var_classes=classes,
    )


class TestExhaustiveDomain(unittest.TestCase):
    N = 240

    def setUp(self):
        self.machine = _machine()

    def test_exact_colorer_matches_brute_force(self):
        agreed = 0
        feasible = 0
        infeasible = 0
        for seed in range(self.N):
            prog = _generate(seed)
            report = liveness_v2(prog)
            vp = build_virtual_program2(prog, self.machine, set(), report)
            status, coloring = exact_color2(vp, self.machine, budget=100_000)
            feas, example, tried = brute_feasibility(
                prog, self.machine, prog.var_classes
            )
            if feas:
                feasible += 1
                self.assertEqual(
                    status, "found",
                    f"seed {seed}: brute force found {example}, solver says "
                    f"{status}",
                )
            else:
                infeasible += 1
                self.assertEqual(
                    status, "infeasible",
                    f"seed {seed}: brute force exhausted {tried} assignments "
                    f"(none valid), solver says {status}",
                )
            agreed += 1
        # the generated family must actually cover BOTH outcomes
        self.assertGreater(feasible, 0)
        self.assertGreater(infeasible, 0)
        self.assertEqual(agreed, self.N)

    def test_pipeline_outcomes_are_consistent_and_verified(self):
        counts = {"allocated_no_spill": 0, "allocated_spill": 0,
                  "infeasible": 0, "budget": 0}
        for seed in range(self.N):
            prog = _generate(seed)
            feas, _, _ = brute_feasibility(prog, self.machine, prog.var_classes)
            result = allocate2(prog, self.machine)
            counts[
                "infeasible" if result.status == "infeasible"
                else "budget" if result.status == "budget_exhausted"
                else "allocated_spill" if result.slots
                else "allocated_no_spill"
            ] += 1
            self.assertNotEqual(result.status, "budget_exhausted")
            if feas:
                self.assertEqual(
                    result.status, "allocated",
                    f"seed {seed}: brute-feasible but solver did not allocate: "
                    f"{result.message}",
                )
                self.assertFalse(
                    result.slots,
                    f"seed {seed}: brute-register-feasible but solver spilled",
                )
            if result.status == "allocated":
                rep = verify_allocation(prog, self.machine, result)
                self.assertTrue(rep.ok, (seed, rep.failures[:3]))
                eq = equivalence_check2(
                    prog, self.machine, result, depth_bound=10, step_budget=2000
                )
                self.assertTrue(eq.equivalent, (seed, eq.failure))
        self.assertGreater(counts["allocated_no_spill"], 0)
        self.assertGreater(counts["infeasible"], 0)

    def test_point_demand_infeasibility_is_proven_not_budget(self):
        # three w-uses at one read point, two w-views: spills cannot help
        prog = Program(
            blocks=(Block(label="e", instrs=(
                Instr(uid=0, uses=(0, 1, 2), defs=()),
            ),),),
            params=(0, 1, 2),
            var_classes={0: "w", 1: "w", 2: "w"},
        )
        feas, _, tried = brute_feasibility(prog, self.machine, prog.var_classes)
        self.assertFalse(feas)
        result = allocate2(prog, self.machine)
        self.assertEqual(result.status, "infeasible")
        self.assertIn("proven infeasible", result.message)

    def test_determinism_across_repeats(self):
        for seed in (0, 7, 99):
            prog = _generate(seed)
            a = allocate2(prog, self.machine)
            b = allocate2(prog, self.machine)
            self.assertEqual(a.status, b.status)
            self.assertEqual(a.assignment, b.assignment)
            self.assertEqual(a.slots, b.slots)


if __name__ == "__main__":
    unittest.main()
