from __future__ import annotations

import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class RoarSample:
    text: str
    label: int


def rerank_tokens(scores: Sequence[float], descending: bool = True) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=descending)


def remove_top_tokens(text: str, ranked_indices: Sequence[int], k: int) -> str:
    tokens = text.split()
    if not tokens:
        return text
    k = max(0, min(k, len(tokens)))
    remove = set(ranked_indices[:k])
    return " ".join(tok for i, tok in enumerate(tokens) if i not in remove)


def remove_top_fraction_tokens(text: str, scores: Sequence[float], remove_fraction: float = 0.1) -> str:
    tokens = text.split()
    if not tokens:
        return text
    if remove_fraction <= 0:
        return text
    remove_k = max(1, int(math.ceil(len(tokens) * remove_fraction)))
    ranked = rerank_tokens(scores)
    return remove_top_tokens(text, ranked, remove_k)


def build_roar_splits(
    samples: Iterable[RoarSample],
    all_scores: Sequence[Sequence[float]],
    remove_k: int | None = None,
    remove_fraction: float = 0.1,
) -> List[RoarSample]:
    output: List[RoarSample] = []
    for sample, scores in zip(samples, all_scores):
        if remove_k is None:
            new_text = remove_top_fraction_tokens(sample.text, scores, remove_fraction=remove_fraction)
        else:
            ranked = rerank_tokens(scores)
            new_text = remove_top_tokens(sample.text, ranked, remove_k)
        output.append(RoarSample(text=new_text, label=sample.label))
    return output


def generate_roar_csv(input_csv: Path, output_csv: Path, all_scores: Sequence[Sequence[float]], remove_fraction: float = 0.1) -> Path:
    try:
        import pandas as pd

        df = pd.read_csv(input_csv)
        if "text" not in df.columns:
            raise ValueError(f"Input CSV missing 'text' column: {input_csv}")
        if len(df) != len(all_scores):
            raise ValueError(f"Scores length ({len(all_scores)}) does not match rows ({len(df)})")

        df = df.copy()
        df["text"] = [remove_top_fraction_tokens(str(text), scores, remove_fraction=remove_fraction) for text, scores in zip(df["text"], all_scores)]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        return output_csv
    except ModuleNotFoundError:
        import csv

        with input_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
            fieldnames = list(rows[0].keys()) if rows else []
        if "text" not in fieldnames:
            raise ValueError(f"Input CSV missing 'text' column: {input_csv}")
        if len(rows) != len(all_scores):
            raise ValueError(f"Scores length ({len(all_scores)}) does not match rows ({len(rows)})")

        for row, scores in zip(rows, all_scores):
            row["text"] = remove_top_fraction_tokens(str(row.get("text", "")), scores, remove_fraction=remove_fraction)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_csv


def run_regular_lm_retrain(
    train_csv: Path,
    test_csv: Path,
    output_json_path: Path,
    model_name: str,
    task: str = "binary",
    script_path: Path = Path("official_baselines/regular_lms_track_ab.py"),
    max_train_rows: int = 5000,
    num_eval_samples: int = 5000,
    num_epochs: int = 2,
    learning_rate: float = 1e-5,
) -> None:
    cmd = [
        sys.executable,
        str(script_path),
        "--model_name",
        model_name,
        "--task",
        task,
        "--train_filepath",
        str(train_csv),
        "--test_filepath",
        str(test_csv),
        "--max_train_rows",
        str(max_train_rows),
        "--num_eval_samples",
        str(num_eval_samples),
        "--num_epochs",
        str(num_epochs),
        "--learning_rate",
        str(learning_rate),
        "--output_json_path",
        str(output_json_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ROAR retraining failed while running: {shlex.join(cmd)}") from exc
