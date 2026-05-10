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
        wrapper = SentimentWrapper("mdeberta", predictor=lambda _text: [1, 2])
        with self.assertRaises(ValueError):
            wrapper.predict("text")

    def test_llama_yes_no_mapping(self):
        wrapper = SentimentWrapper(
            "llama-3",
            predictor=lambda _text: {
                "joy": {"yes": 0.9, "no": 0.1},
                "sadness": (0.2, 0.8),
                "anger": 0.1,
                "fear": 0.1,
                "surprise": 0.2,
                "neutral": 0.5,
            },
        )
        probs = wrapper.predict("text")[0]
        self.assertEqual(len(probs), 6)
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertGreater(probs[0], probs[2])


if __name__ == "__main__":
    unittest.main()
