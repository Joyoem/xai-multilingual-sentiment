from __future__ import annotations

from typing import Iterable, List


def _as_curve(probability_curve: Iterable[float]) -> List[float]:
    return [float(x) for x in probability_curve]


def aopc(probability_curve: Iterable[float]) -> float:
    """Area Over the Perturbation Curve using average confidence drop from p0."""
    curve = _as_curve(probability_curve)
    if len(curve) < 2:
        return 0.0

    p0 = curve[0]
    drops = [max(0.0, p0 - pk) for pk in curve[1:]]
    return sum(drops) / len(drops)


def naopc(probability_curve: Iterable[float], reference_probability: float | None = None) -> float:
    """Normalized AOPC in [0,1] for cross-model/language comparison.

    Defaults to using the final confidence after full perturbation as the
    normalization reference when no explicit reference_probability is provided.
    """
    curve = _as_curve(probability_curve)
    if len(curve) < 2:
        return 0.0

    eps = 1e-12
    denom_ref = float(reference_probability) if reference_probability is not None else curve[-1]
    normalizer = curve[0] - denom_ref
    if normalizer <= eps:
        return 0.0

    score = aopc(curve) / normalizer
    return max(0.0, min(1.0, score))
