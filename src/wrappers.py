from __future__ import annotations

import math
import warnings
from typing import Callable, Iterable, List, Sequence


EmotionVector = List[List[float]]
RawModelOutput = Sequence[float]


class SentimentWrapper:
    """Unified wrapper that returns a (1, 6) emotion probability vector."""

    labels = ("joy", "sadness", "anger", "fear", "surprise", "neutral")

    def __init__(self, backend_name: str, predictor: Callable[[str], RawModelOutput] | None = None):
        self.backend_name = backend_name
        self.predictor = predictor or self._heuristic_predictor

    def predict(self, text: str) -> EmotionVector:
        raw = list(self.predictor(text))
        if len(raw) != 6:
            raise ValueError(f"Expected 6 outputs, got {len(raw)} from {self.backend_name}")
        probs = _normalize_to_probability(raw)
        return [probs]

    def _heuristic_predictor(self, text: str) -> RawModelOutput:
        lower = text.lower()
        logits = [0.0] * 6
        keyword_map = {
            0: ("love", "great", "happy", "excellent"),
            1: ("sad", "cry", "down", "depressed"),
            2: ("angry", "mad", "furious", "hate"),
            3: ("fear", "scared", "afraid", "terrified"),
            4: ("wow", "surprised", "unexpected", "amazing"),
        }
        for idx, words in keyword_map.items():
            logits[idx] += sum(1.0 for word in words if word in lower)
        logits[5] = 1.0
        return logits


def _normalize_to_probability(raw: Iterable[float]) -> List[float]:
    values = list(float(x) for x in raw)
    if not values:
        raise ValueError("Cannot normalize an empty vector")

    min_v = min(values)
    if min_v < 0:
        values = [v - min_v for v in values]

    total = sum(values)
    if total == 0:
        return [1.0 / len(values)] * len(values)

    probs = [v / total for v in values]
    probs_sum = sum(probs)
    if not math.isclose(probs_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        warnings.warn(
            "Probability vector sum deviated from 1.0 due to floating-point precision; corrected final element.",
            RuntimeWarning,
            stacklevel=2,
        )
        probs[-1] = max(0.0, 1.0 - sum(probs[:-1]))
        corrected_sum = sum(probs)
        if any(p < 0.0 or p > 1.0 for p in probs) or not math.isclose(corrected_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Failed to produce a valid probability distribution after correction.")
    return probs
