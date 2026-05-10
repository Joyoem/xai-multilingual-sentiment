from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from src.wrappers import SentimentWrapper


TokenCandidates = Sequence[Tuple[str, float]]
MaskedPredictor = Callable[[str, str, int], TokenCandidates]


def _tokenize(text: str) -> List[str]:
    return text.split()


def _untokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def loo_importance(wrapper: SentimentWrapper, text: str, label_idx: int) -> List[float]:
    tokens = _tokenize(text)
    base = wrapper.predict(text)[0][label_idx]
    scores: List[float] = []

    for i in range(len(tokens)):
        perturbed = _untokenize(tokens[:i] + tokens[i + 1 :])
        perturbed_prob = wrapper.predict(perturbed)[0][label_idx]
        scores.append(base - perturbed_prob)
    return scores


def marginalization_importance(
    wrapper: SentimentWrapper,
    text: str,
    label_idx: int,
    masked_predictor: MaskedPredictor,
    top_k: int = 5,
) -> List[float]:
    """Estimate token importance without OOD masking by sampling plausible replacements."""
    tokens = _tokenize(text)
    base = wrapper.predict(text)[0][label_idx]
    scores: List[float] = []

    for i in range(len(tokens)):
        prefix = _untokenize(tokens[:i])
        suffix = _untokenize(tokens[i + 1 :])
        candidates = masked_predictor(prefix, suffix, top_k)
        if not candidates:
            scores.append(0.0)
            continue

        expected_prob = 0.0
        total_weight = 0.0
        for replacement, weight in candidates:
            if weight <= 0:
                continue
            replaced = tokens[:i] + [replacement] + tokens[i + 1 :]
            expected_prob += weight * wrapper.predict(_untokenize(replaced))[0][label_idx]
            total_weight += weight

        if total_weight == 0:
            scores.append(0.0)
            continue
        expected_prob /= total_weight
        scores.append(base - expected_prob)

    return scores


def lime_importance(wrapper: SentimentWrapper, text: str, label_idx: int) -> List[float]:
    """LIME API entrypoint; currently a LOO-based placeholder (no surrogate model fit)."""
    return loo_importance(wrapper, text, label_idx)


def shap_importance(wrapper: SentimentWrapper, text: str, label_idx: int) -> List[float]:
    """SHAP API entrypoint; currently a LOO-based placeholder (not true Shapley values)."""
    return loo_importance(wrapper, text, label_idx)


def plex_importance(
    wrapper: SentimentWrapper,
    text: str,
    label_idx: int,
    masked_predictor: MaskedPredictor,
    top_k: int = 3,
) -> List[float]:
    """PLEX API entrypoint; currently aliases weighted-token marginalization scoring."""
    return marginalization_importance(wrapper, text, label_idx, masked_predictor=masked_predictor, top_k=top_k)
