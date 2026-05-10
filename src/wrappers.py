from __future__ import annotations

import math
import warnings
from typing import Callable, Iterable, List, Mapping, Sequence


EmotionVector = List[List[float]]
RawModelOutput = Sequence[float] | Mapping[str, object]


class SentimentWrapper:
    """Unified wrapper that always returns a (batch, 6) emotion probability matrix."""

    labels = ("anger", "disgust", "fear", "joy", "sadness", "surprise")

    def __init__(
        self,
        backend_name: str,
        predictor: Callable[[str], RawModelOutput] | None = None,
    ):
        self.backend_name = backend_name
        self.predictor = predictor or self._heuristic_predictor

    def predict(self, text: str | Sequence[str]) -> EmotionVector:
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, Sequence):
            texts = list(text)
            if any(not isinstance(item, str) for item in texts):
                raise TypeError("Sequence inputs to SentimentWrapper.predict must contain only strings.")
        else:
            raise TypeError("SentimentWrapper.predict expects a string or a sequence of strings.")
        if not texts:
            return []

        probs_batch: EmotionVector = []
        for item in texts:
            raw = self.predictor(item)
            probs_batch.append(self._to_probabilities(raw))
        return probs_batch

    def _to_probabilities(self, raw: RawModelOutput) -> List[float]:
        if self._is_llama_style(raw):
            return _normalize_to_probability(self._llama_yesno_to_distribution(raw))

        values = [float(x) for x in list(raw)]  # type: ignore[arg-type]
        if len(values) != len(self.labels):
            raise ValueError(f"Expected {len(self.labels)} outputs, got {len(values)} from {self.backend_name}")

        if any(v < 0.0 or v > 1.0 for v in values):
            values = [_sigmoid(v) for v in values]
        return _normalize_to_probability(values)

    def _is_llama_style(self, raw: RawModelOutput) -> bool:
        if not isinstance(raw, Mapping):
            return False
        if "llama" in self.backend_name.lower():
            return True

        lowered = {str(k).lower(): v for k, v in raw.items()}
        for label in self.labels:
            value = lowered.get(label)
            if isinstance(value, Mapping):
                value_keys = {str(k).lower() for k in value.keys()}
                if "yes" in value_keys or "no" in value_keys or "1" in value_keys or "0" in value_keys:
                    return True
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
                return True
        return False

    def _llama_yesno_to_distribution(self, raw: RawModelOutput) -> List[float]:
        if not isinstance(raw, Mapping):
            raise ValueError("Llama mapping expects dict-style per-label yes/no outputs.")

        output: List[float] = []
        lowered_map = {str(k).lower(): v for k, v in raw.items()}
        for label in self.labels:
            value = lowered_map.get(label, 0.0)
            output.append(_extract_yes_probability(value))
        return output

    def _heuristic_predictor(self, text: str) -> Sequence[float]:
        lower = text.lower()
        logits = [0.0] * 6
        keyword_map = {
            0: ("angry", "mad", "furious", "hate"),
            1: ("disgust", "gross", "nasty", "revolting"),
            2: ("fear", "scared", "afraid", "terrified"),
            3: ("love", "great", "happy", "excellent"),
            4: ("sad", "cry", "down", "depressed"),
            5: ("wow", "surprised", "unexpected", "amazing"),
        }
        for idx, words in keyword_map.items():
            logits[idx] += sum(1.0 for word in words if word in lower)
        return logits


def _extract_yes_probability(value: object) -> float:
    if isinstance(value, Mapping):
        lowered = {str(k).lower(): v for k, v in value.items()}
        yes = float(lowered.get("yes", lowered.get("1", 0.0)))
        inferred_no = 1.0 - yes
        no = float(lowered.get("no", lowered.get("0", inferred_no)))
        denom = yes + no
        return yes / denom if denom > 0 else 0.0

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        arr = list(value)
        if len(arr) == 2:
            yes = float(arr[0])
            no = float(arr[1])
            denom = yes + no
            return yes / denom if denom > 0 else 0.0
        if len(arr) == 1:
            return float(arr[0])

    return float(value)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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
