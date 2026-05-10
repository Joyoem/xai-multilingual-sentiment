from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class RoarSample:
    text: str
    label: int


def rerank_tokens(scores: Sequence[float], descending: bool = True) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=descending)


def remove_top_tokens(text: str, ranked_indices: Sequence[int], k: int) -> str:
    tokens = text.split()
    remove = set(ranked_indices[:k])
    return " ".join(tok for i, tok in enumerate(tokens) if i not in remove)


def build_roar_splits(samples: Iterable[RoarSample], all_scores: Sequence[Sequence[float]], remove_k: int) -> List[RoarSample]:
    output: List[RoarSample] = []
    for sample, scores in zip(samples, all_scores):
        ranked = rerank_tokens(scores)
        output.append(RoarSample(text=remove_top_tokens(sample.text, ranked, remove_k), label=sample.label))
    return output
