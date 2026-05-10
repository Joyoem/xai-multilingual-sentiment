from __future__ import annotations

import warnings
from typing import Callable, List, Sequence, Tuple

from src.wrappers import SentimentWrapper


TokenCandidates = Sequence[Tuple[str, float]]
MaskedPredictor = Callable[[str, str, int], TokenCandidates]


def _tokenize(text: str) -> List[str]:
    return text.split()


def _untokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


def loo_importance(wrapper: SentimentWrapper, text: str, label_idx: int) -> List[float]:
    tokens = _tokenize(text)
    if not tokens:
        return []

    base = wrapper.predict(text)[0][label_idx]
    scores: List[float] = []

    for i in range(len(tokens)):
        perturbed = _untokenize(tokens[:i] + tokens[i + 1 :])
        perturbed_prob = wrapper.predict(perturbed)[0][label_idx]
        scores.append(base - perturbed_prob)
    return scores


def build_mlm_masked_predictor(model_name: str = "bert-base-multilingual-cased") -> MaskedPredictor:
    """Create a masked-token predictor used by input marginalization.

    Implements p(w|context) with a multilingual masked-LM.
    """

    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError(
            "transformers and torch are required for MLM marginalization predictor "
            "(install with: pip install transformers torch)."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError(f"Model '{model_name}' does not define a mask token.")

    def predict(prefix: str, suffix: str, k: int) -> TokenCandidates:
        masked_text = _untokenize([piece for piece in (prefix, tokenizer.mask_token, suffix) if piece])
        encoded = tokenizer(masked_text, return_tensors="pt")
        input_ids = encoded["input_ids"][0]
        mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            return []

        mask_pos = int(mask_positions[0].item())
        with torch.no_grad():
            logits = model(**encoded).logits[0, mask_pos]
            probs = torch.softmax(logits, dim=-1)

        top_k = max(1, min(int(k), probs.shape[-1]))
        values, indices = torch.topk(probs, k=top_k)

        candidates: List[Tuple[str, float]] = []
        for token_id, prob in zip(indices.tolist(), values.tolist()):
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            word = tokenizer.convert_tokens_to_string([token]).strip()
            if not word:
                continue
            candidates.append((word, float(prob)))
        return candidates

    return predict


def marginalization_importance(
    wrapper: SentimentWrapper,
    text: str,
    label_idx: int,
    masked_predictor: MaskedPredictor,
    top_k: int = 5,
) -> List[float]:
    """Input marginalization attribution following Eq. p(y|x\\w_i)=sum_w p(w|ctx)p(y|x[w_i->w])."""
    tokens = _tokenize(text)
    if not tokens:
        return []

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
            expected_prob += float(weight) * wrapper.predict(_untokenize(replaced))[0][label_idx]
            total_weight += float(weight)

        if total_weight <= 0:
            scores.append(0.0)
            continue
        expected_prob /= total_weight
        scores.append(base - expected_prob)

    return scores


def lime_importance(wrapper: SentimentWrapper, text: str, label_idx: int) -> List[float]:
    """LIME integration with graceful fallback to LOO if lime is unavailable."""
    tokens = _tokenize(text)
    if not tokens:
        return []

    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return loo_importance(wrapper, text, label_idx)

    explainer = LimeTextExplainer(class_names=list(wrapper.labels), split_expression=r"\s+")

    def classifier_fn(texts: Sequence[str]) -> list[list[float]]:
        return wrapper.predict(list(texts))

    try:
        explanation = explainer.explain_instance(
            text,
            classifier_fn,
            labels=(label_idx,),
            num_features=len(tokens),
        )
        weights = dict(explanation.as_list(label=label_idx))
        return [float(weights.get(tok, 0.0)) for tok in tokens]
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:  # pragma: no cover - optional backend behavior
        warnings.warn(f"LIME failed ({exc!r}); falling back to LOO scores.", RuntimeWarning, stacklevel=2)
        return loo_importance(wrapper, text, label_idx)


def shap_importance(wrapper: SentimentWrapper, text: str, label_idx: int) -> List[float]:
    """SHAP integration with graceful fallback to LOO if shap is unavailable."""
    tokens = _tokenize(text)
    if not tokens:
        return []

    try:
        import shap
    except ImportError:
        return loo_importance(wrapper, text, label_idx)

    def model_fn(texts: Sequence[str]) -> list[list[float]]:
        return wrapper.predict(list(texts))

    try:
        masker = shap.maskers.Text(r"\W+")
        explainer = shap.Explainer(model_fn, masker)
        values = explainer([text])
        token_values = values.values[0]
        if token_values.ndim == 2:
            class_values = token_values[:, label_idx]
        else:
            class_values = token_values
        class_values_list = [float(v) for v in class_values]
        if len(class_values_list) == len(tokens):
            return class_values_list
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:  # pragma: no cover - optional backend behavior
        warnings.warn(f"SHAP failed ({exc!r}); falling back to LOO scores.", RuntimeWarning, stacklevel=2)

    return loo_importance(wrapper, text, label_idx)


def plex_importance(
    wrapper: SentimentWrapper,
    text: str,
    label_idx: int,
    masked_predictor: MaskedPredictor,
    top_k: int = 3,
) -> List[float]:
    """Perturbation-light proxy: equal-weight blend of LOO and low-k marginalization for lower query cost."""
    loo_scores = loo_importance(wrapper, text, label_idx)
    marg_scores = marginalization_importance(wrapper, text, label_idx, masked_predictor=masked_predictor, top_k=top_k)
    if not loo_scores:
        return marg_scores
    if len(loo_scores) != len(marg_scores):
        warnings.warn("PLEX score-length mismatch detected; returning LOO scores.", RuntimeWarning, stacklevel=2)
        return loo_scores
    return [(l + m) / 2.0 for l, m in zip(loo_scores, marg_scores)]
