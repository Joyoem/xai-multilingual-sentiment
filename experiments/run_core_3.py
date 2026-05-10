from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Sequence

import pandas as pd

from src.explainers import (
    MaskedPredictor,
    build_mlm_masked_predictor,
    lime_importance,
    loo_importance,
    marginalization_importance,
    plex_importance,
    shap_importance,
)
from src.metrics import aopc, naopc
from src.wrappers import SentimentWrapper


LANGS: Sequence[str] = ("eng", "afr", "jav")


def _default_masked_predictor(prefix: str, suffix: str, _k: int) -> list[tuple[str, float]]:
    del prefix, suffix
    return [("the", 1.0)]


def _remove_top_k_tokens(text: str, scores: Sequence[float], k: int) -> str:
    tokens = text.split()
    if not tokens:
        return text
    k = max(0, min(k, len(tokens)))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    remove = set(ranked[:k])
    return " ".join(tok for i, tok in enumerate(tokens) if i not in remove)


def _curve_from_scores(wrapper: SentimentWrapper, text: str, label_idx: int, scores: Sequence[float]) -> List[float]:
    tokens = text.split()
    if not tokens:
        return [wrapper.predict(text)[0][label_idx]]

    curve = [wrapper.predict(text)[0][label_idx]]
    for k in range(1, len(tokens) + 1):
        perturbed = _remove_top_k_tokens(text, scores, k)
        curve.append(wrapper.predict(perturbed)[0][label_idx])
    return curve


def _sample_rows(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()
    rng = random.Random(seed)
    indices = rng.sample(list(df.index), sample_size)
    return df.loc[indices].reset_index(drop=True)


def _build_method_scores(
    wrapper: SentimentWrapper,
    text: str,
    label_idx: int,
    masked_predictor: MaskedPredictor,
) -> dict[str, List[float]]:
    return {
        "loo": loo_importance(wrapper, text, label_idx),
        "shap": shap_importance(wrapper, text, label_idx),
        "lime": lime_importance(wrapper, text, label_idx),
        "marginalization": marginalization_importance(wrapper, text, label_idx, masked_predictor=masked_predictor),
        "plex": plex_importance(wrapper, text, label_idx, masked_predictor=masked_predictor),
    }


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    use_mlm: bool,
    mlm_model: str,
) -> pd.DataFrame:
    wrapper = SentimentWrapper("mdeberta")
    masked_predictor: MaskedPredictor = _default_masked_predictor
    if use_mlm:
        masked_predictor = build_mlm_masked_predictor(mlm_model)

    records: list[dict[str, object]] = []

    for lang_idx, lang in enumerate(LANGS):
        path = data_root / f"{lang}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "text" not in df.columns:
            raise ValueError(f"Missing text column in {path}")

        sampled = _sample_rows(df, sample_size=sample_size, seed=seed + lang_idx)
        for row_idx, row in sampled.iterrows():
            text = str(row["text"])
            base_probs = wrapper.predict(text)[0]
            label_idx = int(base_probs.index(max(base_probs)))

            method_scores = _build_method_scores(wrapper, text, label_idx, masked_predictor)
            for method, scores in method_scores.items():
                curve = _curve_from_scores(wrapper, text, label_idx, scores)
                records.append(
                    {
                        "language": lang,
                        "row_index": int(row_idx),
                        "method": method,
                        "label_idx": label_idx,
                        "text": text,
                        "token_count": len(text.split()),
                        "scores": json.dumps(scores),
                        "aopc": aopc(curve),
                        "naopc": naopc(curve),
                    }
                )

    out_df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "core3_results.csv"
    out_json = output_dir / "core3_results.json"
    out_df.to_csv(out_csv, index=False)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "languages": list(LANGS),
        "use_mlm": use_mlm,
        "mlm_model": mlm_model,
        "records": records,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3-language XAI core experiments (eng/afr/jav).")
    parser.add_argument("--data-root", type=Path, default=Path("data/track_a"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlm-model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--disable-mlm", action="store_true", help="Use deterministic fallback instead of masked LM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        use_mlm=not args.disable_mlm,
        mlm_model=args.mlm_model,
    )
    print(f"Saved {len(df)} records to {args.output_dir}")


if __name__ == "__main__":
    main()
