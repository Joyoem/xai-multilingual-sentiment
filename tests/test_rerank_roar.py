import csv
import tempfile
import unittest
from pathlib import Path

from src.rerank_roar import generate_roar_csv, remove_top_fraction_tokens


class TestRerankRoar(unittest.TestCase):
    def test_remove_top_fraction_tokens(self):
        text = "a b c d e"
        scores = [0.1, 0.2, 0.9, 0.3, 0.4]
        out = remove_top_fraction_tokens(text, scores, remove_fraction=0.2)
        self.assertEqual(out, "a b d e")

    def test_generate_roar_csv(self):
        with tempfile.TemporaryDirectory() as d:
            input_csv = Path(d) / "input.csv"
            output_csv = Path(d) / "output.csv"
            with input_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["text", "joy", "sadness", "anger", "fear", "surprise", "neutral"],
                )
                writer.writeheader()
                writer.writerow({"text": "a b c", "joy": 1, "sadness": 0, "anger": 0, "fear": 0, "surprise": 0, "neutral": 0})
                writer.writerow({"text": "d e f", "joy": 0, "sadness": 1, "anger": 0, "fear": 0, "surprise": 0, "neutral": 0})

            generate_roar_csv(input_csv, output_csv, all_scores=[[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]], remove_fraction=0.34)
            with output_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["text"], "c")
            self.assertEqual(rows[1]["text"], "f")


if __name__ == "__main__":
    unittest.main()
