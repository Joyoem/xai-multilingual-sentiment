import unittest

from src.explainers import loo_importance, marginalization_importance
from src.metrics import aopc, naopc
from src.wrappers import SentimentWrapper


class TestExplainersAndMetrics(unittest.TestCase):
    def test_loo_returns_score_per_token(self):
        wrapper = SentimentWrapper("toy", predictor=lambda text: [text.count("good") + 1, 1, 1, 1, 1, 1])
        scores = loo_importance(wrapper, "good movie", 0)
        self.assertEqual(len(scores), 2)

    def test_marginalization_runs_with_weighted_candidates(self):
        wrapper = SentimentWrapper("toy", predictor=lambda text: [text.count("great") + 1, 1, 1, 1, 1, 1])

        def predictor(_prefix, _suffix, _k):
            return [("great", 0.7), ("okay", 0.3)]

        scores = marginalization_importance(wrapper, "very film", 0, predictor)
        self.assertEqual(len(scores), 2)

    def test_aopc_and_naopc(self):
        curve = [0.9, 0.6, 0.3]
        self.assertAlmostEqual(aopc(curve), 0.45)
        self.assertAlmostEqual(naopc(curve), 0.75)

    def test_naopc_zero_or_negative_normalizer(self):
        curve = [0.4, 0.4, 0.5]
        self.assertEqual(naopc(curve, reference_probability=0.4), 0.0)


if __name__ == "__main__":
    unittest.main()
