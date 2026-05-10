from pathlib import Path


if __name__ == "__main__":
    langs = ["eng", "afr", "jav"]
    data_root = Path("data/track_a")
    for lang in langs:
        print(f"Prepared core-run placeholder for {data_root / f'{lang}.csv'}")
