import unittest

from ap_upbc.core.evidence import Evidence
from ap_upbc.inference.engine import APUPBCEngine
from ap_upbc.inference.pruning import Pruner
from ap_upbc.inference.scheduler import DynamicHeapMapper


class TestEvidence(unittest.TestCase):
    def test_utility_index_formula(self):
        # U(E) = max(|LS - 1|, |1 - LN|)
        self.assertAlmostEqual(Evidence("A", ls=5.0, ln=0.8).utility, 4.0)
        self.assertAlmostEqual(Evidence("B", ls=0.5, ln=0.1).utility, 0.9)
        self.assertAlmostEqual(Evidence("C", ls=1.0, ln=1.0).utility, 0.0)

    def test_invalid_likelihood_ratio_rejected(self):
        with self.assertRaises(ValueError):
            Evidence("bad", ls=-1.0, ln=0.5)
        with self.assertRaises(ValueError):
            Evidence("bad", ls=2.0, ln=0.0)


class TestDynamicHeapMapper(unittest.TestCase):
    def test_pops_in_descending_utility_order(self):
        mapper = DynamicHeapMapper()
        low = Evidence("low", ls=1.2, ln=0.95)     # U = 0.2
        high = Evidence("high", ls=8.0, ln=0.5)    # U = 7.0
        mid = Evidence("mid", ls=5.0, ln=0.8)      # U = 4.0
        for ev in (low, high, mid):
            mapper.push_frontier(ev)

        self.assertEqual(mapper.pop_highest_utility().name, "high")
        self.assertEqual(mapper.pop_highest_utility().name, "mid")
        self.assertEqual(mapper.pop_highest_utility().name, "low")
        self.assertIsNone(mapper.pop_highest_utility())

    def test_equal_utility_preserves_insertion_order(self):
        mapper = DynamicHeapMapper()
        first = Evidence("first", ls=3.0, ln=0.9)
        second = Evidence("second", ls=3.0, ln=0.9)
        mapper.push_frontier(first)
        mapper.push_frontier(second)
        self.assertEqual(mapper.pop_highest_utility().name, "first")


class TestPruner(unittest.TestCase):
    def test_omax_uses_only_favorable_ratios(self):
        # LS < 1 인 증거는 1로 처리된다
        pruner = Pruner(threshold_tau=10.0)
        remaining = [Evidence("A", ls=2.0, ln=0.5), Evidence("B", ls=0.5, ln=0.1)]
        self.assertAlmostEqual(pruner.calculate_omax(3.0, remaining), 6.0)

    def test_should_prune_below_threshold(self):
        pruner = Pruner(threshold_tau=10.0)
        remaining = [Evidence("A", ls=2.0, ln=0.5)]
        self.assertTrue(pruner.should_prune(1.0, remaining))    # O_max = 2 < 10
        self.assertFalse(pruner.should_prune(6.0, remaining))   # O_max = 12 >= 10


class TestAPUPBCEngine(unittest.TestCase):
    def _knowledge_base(self):
        return [
            Evidence("고열", ls=5.0, ln=0.8),
            Evidence("여행력", ls=8.0, ln=0.5),
            Evidence("항체", ls=0.5, ln=0.1),
        ]

    def test_all_confirmed_reaches_early_acceptance(self):
        engine = APUPBCEngine(prior_odds=1.0, tau=15.0, verbose=False)
        engine.load_frontier(self._knowledge_base())
        result = engine.run_inference(answer_provider=lambda ev: True)

        # 여행력(8.0), 고열(5.0) 확인 시 odds 40 >= 15 이므로 항체 질의 전에 채택
        self.assertEqual(result.status, "ACCEPTED")
        self.assertEqual(result.queries_asked, 2)
        self.assertEqual(result.query_log, ["여행력", "고열"])
        self.assertAlmostEqual(result.final_odds, 40.0)

    def test_denied_evidence_applies_ln_and_prunes(self):
        engine = APUPBCEngine(prior_odds=1.0, tau=15.0, verbose=False)
        engine.load_frontier(self._knowledge_base())
        result = engine.run_inference(answer_provider=lambda ev: False)

        # 여행력 부정(LN=0.5) 후 O_max = 0.5 * 5.0 * 1.0 = 2.5 < 15
        self.assertEqual(result.status, "PRUNED")
        self.assertEqual(result.queries_asked, 1)
        self.assertAlmostEqual(result.final_odds, 0.5)

    def test_invalid_prior_odds_rejected(self):
        with self.assertRaises(ValueError):
            APUPBCEngine(prior_odds=0.0, tau=15.0)


if __name__ == "__main__":
    unittest.main()
