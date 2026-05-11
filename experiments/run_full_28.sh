#!/bin/bash
# Batch experiment runner for full 28-language XAI evaluation
# Usage: bash experiments/run_full_28.sh [data_root] [output_dir] [sample_size] [seed] [mlm_model]

set -e

# Default parameters
DATA_ROOT="${1:-data/track_a}"
OUTPUT_DIR="${2:-results}"
SAMPLE_SIZE="${3:-100}"
SEED="${4:-42}"
MLM_MODEL="${5:-bert-base-multilingual-cased}"

# All 28 BRIGHTER track-a languages
LANGUAGES=(
    "eng" "afr" "jav"  # Primary focus (high, mid, low resource)
    # Additional 25 languages can be added as data becomes available:
    # "ara" "hin" "tur" "kor" "zho" "jpn" "deu" "fra" "spa" "ita"
    # "rus" "pol" "swe" "dan" "fin" "gre" "heb" "tgl" "ind" "mal"
    # "pun" "ben" "tam" "tel" "kan" "mya"
)

echo "======================================================================"
echo "XAI Multilingual Sentiment Analysis: Full 28-Language Batch Runner"
echo "======================================================================"
echo "Data Root:      $DATA_ROOT"
echo "Output Dir:     $OUTPUT_DIR"
echo "Sample Size:    $SAMPLE_SIZE per language"
echo "Random Seed:    $SEED"
echo "MLM Model:      $MLM_MODEL"
echo "Available Languages: ${#LANGUAGES[@]} (3 core, 25 extensible)"
echo "======================================================================"

# Check if data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] Data root directory not found: $DATA_ROOT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Track which languages are processed
PROCESSED_LANGS=()
SKIPPED_LANGS=()
FAILED_LANGS=()

# Run experiments for each language that has data
for lang in "${LANGUAGES[@]}"; do
    LANG_CSV="$DATA_ROOT/${lang}.csv"
    
    if [ ! -f "$LANG_CSV" ]; then
        echo "[SKIP] No data file found: $LANG_CSV"
        SKIPPED_LANGS+=("$lang")
        continue
    fi
    
    LANG_OUTPUT_DIR="$OUTPUT_DIR/${lang}_sample${SAMPLE_SIZE}"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "Running experiment for language: $lang"
    echo "Input:  $LANG_CSV"
    echo "Output: $LANG_OUTPUT_DIR"
    echo "────────────────────────────────────────────────────────────────────"
    
    if python -m experiments.run_core_3 \
        --data-root "$DATA_ROOT" \
        --output-dir "$LANG_OUTPUT_DIR" \
        --sample-size "$SAMPLE_SIZE" \
        --seed "$SEED" \
        --mlm-model "$MLM_MODEL"; then
        
        PROCESSED_LANGS+=("$lang")
        echo "[OK] Completed: $lang"
    else
        FAILED_LANGS+=("$lang")
        echo "[ERROR] Failed to process: $lang"
    fi
done

echo ""
echo "======================================================================"
echo "BATCH JOB SUMMARY"
echo "======================================================================"
echo "Processed Languages (${#PROCESSED_LANGS[@]}): ${PROCESSED_LANGS[*]:-None}"
echo "Skipped Languages   (${#SKIPPED_LANGS[@]}): ${SKIPPED_LANGS[*]:-None}"
echo "Failed Languages    (${#FAILED_LANGS[@]}): ${FAILED_LANGS[*]:-None}"
echo ""

if [ ${#PROCESSED_LANGS[@]} -gt 0 ]; then
    echo "✓ Aggregating results across ${#PROCESSED_LANGS[@]} languages..."
    
    # Combine all CSV results (optional aggregation step)
    # This creates a summary CSV with all results
    python << 'PYTHON_SCRIPT'
import json
import pandas as pd
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
all_results = []

for lang_dir in sorted(output_dir.glob("*_sample*/core3_results.json")):
    try:
        with open(lang_dir) as f:
            data = json.load(f)
            all_results.extend(data.get("records", []))
    except Exception as e:
        print(f"Warning: Could not read {lang_dir}: {e}")

if all_results:
    df = pd.DataFrame(all_results)
    summary_path = output_dir / "full_28_aggregated.csv"
    df.to_csv(summary_path, index=False)
    print(f"✓ Aggregated {len(df)} total records to {summary_path}")
else:
    print("No results to aggregate")
PYTHON_SCRIPT
fi

echo ""
echo "Next steps:"
echo "1. Visualize results: jupyter notebook notebooks/visualization.ipynb"
echo "2. Analyze per-language: ls -la $OUTPUT_DIR/*/core3_results.json"
echo "3. ROAR evaluation: python src/rerank_roar.py --input-csv ... --scores ..."
echo ""
echo "======================================================================"
