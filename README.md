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
├── data/                   # 存放 BRIGHTER 数据集的子集
│   ├── track_a/            # 类别标签数据 (Categorical)
│   │   ├── eng.csv         # 高资源基准 (High)
│   │   ├── afr.csv         # 中资源对比 (Mid)
│   │   └── jav.csv         # 零资源挑战 (Zero/Low)
│
├── official_baselines/     # 你下载的那 3 个官方脚本
│   ├── llms.py             # 处理 LLM 推理逻辑
│   ├── regular_lms_track_ab.py # 处理 BERT 类模型逻辑
│   └── process_llm_results.py
│
├── src/                    # 你的核心实证研究代码 (My Engagement)
│   ├── wrappers.py         # 统一 mDeBERTa 和 Llama 3 概率输出的接口
│   ├── explainers.py       # 5 种方法的实现 (SHAP, LIME, LOO, Marginalization, PLEX)
│   ├── metrics.py          # AOPC, NAOPC 指标的数学实现
│   └── rerank_roar.py      # ROAR 重训练逻辑实现 [cite: 1158-1165]
│
├── experiments/            # 自动化运行脚本
│   ├── run_full_28.sh      # 28 语种全量扫描 (PLEX 效率证明)
│   └── run_core_3.py       # 3 核心语种深度对比 (OOD 本质探究)
│
├── notebooks/              # 用于可视化和 Case Study
│   └── visualization.ipynb # 绘制热力图与 AOPC 曲线 (解决 Slide 9/13 痛点)
│
├── results/                # 实验产出的 JSON/CSV 数据
├── README.md               # 项目介绍、安装说明、实验结论
└── requirements.txt        # 环境依赖 (Captum, Transformers, vLLM 等)

```
