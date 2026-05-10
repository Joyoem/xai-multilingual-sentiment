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
- AOPC
- NAOPC (normalized AOPC for cross-model comparability)

## Key Findings (Template)
Use your experiments under `experiments/` and plots from `notebooks/visualization.ipynb` to report findings, e.g.:
- In low-resource settings (e.g. Javanese), NAOPC can drop sharply, showing explanation instability.
- Marginalization often gives smoother degradation curves than hard-mask occlusion.

## Repository Layout
```text
.
├── data/
├── official_baselines/
├── src/
├── experiments/
├── notebooks/
├── results/
├── tests/
└── requirements.txt
```
