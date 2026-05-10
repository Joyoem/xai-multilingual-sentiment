from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence


LANGUAGES: Sequence[str] = ("eng", "afr", "jav")
EMOTION_COLUMNS: Sequence[str] = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
TEXT_KEYS: Sequence[str] = ("text", "sentence", "content")
LANG_KEYS: Sequence[str] = ("lang", "language", "iso", "iso_lang")


def _select_key(row: Mapping[str, object], keys: Iterable[str]) -> str | None:
    lowered = {k.lower(): k for k in row.keys()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def _coerce_binary(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _row_to_track_a_record(row: Mapping[str, object], text_key: str) -> dict[str, object]:
    record: dict[str, object] = {"text": str(row.get(text_key, "")).strip()}
    for emotion in EMOTION_COLUMNS:
        record[emotion] = _coerce_binary(row.get(emotion, 0))
    return record


def prepare_track_a(
    dataset_name: str,
    split: str,
    output_dir: Path,
    languages: Sequence[str] = LANGUAGES,
) -> dict[str, int]:
    """Fetch BRIGHTER from Hugging Face datasets and export Track A CSV slices."""
    from datasets import load_dataset  # lazy import so tests don't require datasets at import time
    import pandas as pd

    dataset = load_dataset(dataset_name, split=split)
    if len(dataset) == 0:
        raise ValueError(f"Empty split '{split}' from dataset '{dataset_name}'.")

    sample = dataset[0]
    if not isinstance(sample, Mapping):
        raise ValueError("Dataset rows must be mapping-like records.")

    text_key = _select_key(sample, TEXT_KEYS)
    lang_key = _select_key(sample, LANG_KEYS)
    if text_key is None or lang_key is None:
        raise ValueError(
            "Could not find required text/language keys in dataset rows. "
            f"Available keys: {list(sample.keys())}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for lang in languages:
        lang_rows = []
        for row in dataset:
            if str(row.get(lang_key, "")).strip().lower() != lang:
                continue
            lang_rows.append(_row_to_track_a_record(row, text_key=text_key))

        if not lang_rows:
            raise ValueError(f"No rows found for language='{lang}' in split '{split}'.")

        df = pd.DataFrame(lang_rows, columns=["text", *EMOTION_COLUMNS])
        out_path = output_dir / f"{lang}.csv"
        df.to_csv(out_path, index=False)
        counts[lang] = len(df)

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BRIGHTER Track-A CSV files for core 3-language XAI runs.")
    parser.add_argument("--dataset-name", type=str, default="Joyoem/BRIGHTER", help="Hugging Face dataset id")
    parser.add_argument("--split", type=str, default="test", help="Dataset split name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/track_a"),
        help="Directory where eng.csv / afr.csv / jav.csv are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = prepare_track_a(args.dataset_name, args.split, args.output_dir)
    for lang, count in counts.items():
        print(f"Wrote {args.output_dir / f'{lang}.csv'} ({count} rows)")


if __name__ == "__main__":
    main()
