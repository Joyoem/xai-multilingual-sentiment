# XAI-Multilingual-Benchmark

## Project Objective
This project studies occlusion/perturbation-based explainability for multilingual sentiment and emotion analysis, with a focus on whether **marginalization** can reduce out-of-distribution (OOD) artifacts introduced by hard masking.

## Dataset
We use BRIGHTER Track A style data organization and provide a starter subset in `data/track_a`:
- `eng.csv` (high-resource baseline)
- `afr.csv` (mid-resource comparison)
- `jav.csv` (low/zero-resource challenge)

## Methods
Core methods are implemented in `src/explainers.py`:
- LOO (Leave-One-Out)
- LIME-style local perturbation approximation
- SHAP-style additive approximation
- Marginalization-based replacement (context-aware perturbation)
- PLEX-style perturbation-light proxy

Evaluation metrics are in `src/metrics.py`:
- NAOPC (normalized AOPC for cross-model comparability)

## Key Findings (Template)
Use your experiments under `experiments/` and plots from `notebooks/analysis.ipynb` to report findings, e.g.:
- In low-resource settings (e.g. Javanese), NAOPC can drop sharply, showing explanation instability.
- Marginalization often gives smoother degradation curves than hard-mask occlusion.

## Repository Layout
```text
.
├── data/                   
│   ├── track_a/            # Categorical
│   │   ├── eng.csv         # High
│   │   ├── afr.csv         # Mid
│   │   └── jav.csv         # Low
│
├── official_baselines/     # official scripts
│   ├── llms.py             # 
│   ├── regular_lms_track_ab.py # 
│   └── process_llm_results.py
│
├── src/                    # My Engagement
│   ├── wrappers.py         # 
│   ├── explainers.py       # SHAP, LIME, LOO, Marginalization, PLEX
│   ├── metrics.py          # NAOPC 
│   └── rerank_roar.py      # ROAR （not included in paper） 
│
├── experiments/            # run scripts
│   ├── run_full_28.sh      # 28 - PLEX （limitd to cost，no implement）
│   └── run_core_3.py       # 3 core experiment
│
├── notebooks/              # colab/kaggle environment
│   └── analysis.ipynb      # results visuaization 
│
├── results/               
├── README.md               
└── requirements.txt       

```
