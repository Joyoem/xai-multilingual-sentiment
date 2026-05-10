from __future__ import annotations

from typing import Iterable, List


def aopc(probability_curve: Iterable[float]) -> float:
    """Area Over the Perturbation Curve.

    The first element should be the original confidence p0, followed by p1..pk.
    """
    curve = list(probability_curve)
    if len(curve) < 2:
        return 0.0

    p0 = curve[0]
    drops = [p0 - pk for pk in curve[1:]]
    return sum(drops) / len(drops)


def naopc(probability_curve: Iterable[float], reference_probability: float | None = None) -> float:
    """Normalized AOPC for cross-model/language comparability."""
    curve: List[float] = list(probability_curve)
    if len(curve) < 2:
        return 0.0

    denom_ref = reference_probability if reference_probability is not None else min(curve)
    normalizer = curve[0] - denom_ref
    if normalizer <= 0:
        return 0.0
    return aopc(curve) / normalizer
