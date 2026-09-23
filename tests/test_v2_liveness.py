"""Hand-computed liveness expectations for exceptional control flow (v2).

Expected sets were derived on paper from the documented equations, not by
calling the allocator.
"""

import unittest

from regalloc.liveness import liveness_v2
from regalloc.model import Block, Instr, PhiNode, Program


def _throwing():
    """::

        b0: v1 = f(v0); v2 = call(v1) may_throw  -> b1, ~~> b2(phi v4<-v1)
        b1: v3 = g(v2)
        b2: sink(v4)
    """
    b0 = Block(
        label="b0",
        instrs=(
            Instr(uid=0, uses=(0,), defs=(1,)),
            Instr(uid=1, uses=(1,), defs=(2,), is_call=True, may_throw=True),
        ),
        succs=(1,),
        exc_succs=(2,),
    )
    b1 = Block(
        label="b1",
        instrs=(Instr(uid=2, uses=(2,), defs=(3,)),),
        succs=(),
    )
    b2 = Block(
        label="handler",
        instrs=(Instr(uid=3, uses=(4,), defs=(), note="sink"),),
        succs=(),
        phis=(PhiNode(def_var=4, sources=((0, 1),)),),
    )
    return Program(
        blocks=(b0, b1, b2),
        params=(0,),
        name="throw",
        var_classes={0: "w", 1: "w", 2: "w", 3: "w", 4: "w"},
    )


class TestExceptionalLiveness(unittest.TestCase):
    def setUp(self):
        self.rep = liveness_v2(_throwing())

    def test_normal_edge_carries_result(self):
        # normal completion: v2 exists and feeds b1
        self.assertEqual(set(self.rep.edge_live[(0, 1)]), {2})

    def test_exceptional_edge_carries_prethrow_value(self):
        # exceptional completion at uid1: v2 does NOT exist; the handler
        # reads v1 (via its phi source), so v1 must survive to the edge
        self.assertEqual(set(self.rep.edge_live[(0, 2)]), {1})

    def test_throw_result_not_live_on_exceptional_edge(self):
        self.assertNotIn(2, self.rep.edge_live[(0, 2)])

    def test_instr_after_is_union_of_continuations(self):
        # after uid1: v2 on the normal continuation, v1 on the exceptional
        self.assertEqual(set(self.rep.instr_after[0][1]), {1, 2})

    def test_instr_before_throw(self):
        self.assertEqual(set(self.rep.instr_before[0][1]), {1})

    def test_value_needed_by_handler_is_live_in(self):
        # v0 (param) is used by uid0 -> live at entry; v1 is defined by
        # uid0, so it is NOT in live-in but IS live before uid1 (the throw)
        self.assertEqual(set(self.rep.live_in[0]), {0})
        self.assertIn(1, self.rep.instr_before[0][1])

    def test_handler_livein_phi_boundary(self):
        self.assertEqual(set(self.rep.live_in[2]), set())
        self.assertEqual(set(self.rep.instr_before[2][0]), {4})


class TestThrowNotLastInstr(unittest.TestCase):
    def test_instructions_after_throw_are_normal_only(self):
        """::

            b0: v1 = op; v2 = op may_throw (~~> b2); v3 = f(v2)
            b1 normal exit reachable via fallthrough; b2 uses v1
        """
        b0 = Block(
            label="b0",
            instrs=(
                Instr(uid=0, uses=(), defs=(1,)),
                Instr(uid=1, uses=(1,), defs=(2,), may_throw=True),
                Instr(uid=2, uses=(2,), defs=(3,)),
            ),
            succs=(),
            exc_succs=(1,),
        )
        b1 = Block(
            label="handler",
            instrs=(Instr(uid=3, uses=(1,), defs=(), note="sink v1"),),
            succs=(),
        )
        prog = Program(
            blocks=(b0, b1),
            var_classes={1: "w", 2: "w", 3: "w"},
        )
        rep = liveness_v2(prog)
        # exceptional edge at uid1 carries v1, not v2/v3
        self.assertEqual(set(rep.edge_live[(0, 1)]), {1})
        # uid2 (after the throw) sees v2 only on the normal path
        self.assertEqual(set(rep.instr_before[0][2]), {2})
        # after uid1 the may-set includes v1 (handler) and v2 (normal)
        self.assertEqual(set(rep.instr_after[0][1]), {1, 2})


class TestDeterminism(unittest.TestCase):
    def test_repeated_runs_equal(self):
        a = liveness_v2(_throwing())
        b = liveness_v2(_throwing())
        self.assertEqual(a.as_jsonable(), b.as_jsonable())
