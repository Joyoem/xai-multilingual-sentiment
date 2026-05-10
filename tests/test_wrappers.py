import unittest

from src.wrappers import SentimentWrapper


class TestSentimentWrapper(unittest.TestCase):
    def test_predict_shape_and_normalization(self):
        wrapper = SentimentWrapper("mdeberta", predictor=lambda _text: [2, 1, 0, 0, 0, 1])
        probs = wrapper.predict("I love this")
        self.assertEqual(len(probs), 1)
        self.assertEqual(len(probs[0]), 6)
        self.assertAlmostEqual(sum(probs[0]), 1.0)

    def test_predict_rejects_wrong_dim(self):
        wrapper = SentimentWrapper("llama", predictor=lambda _text: [1, 2])
        with self.assertRaises(ValueError):
            wrapper.predict("text")


if __name__ == "__main__":
    unittest.main()
