[![GitHub Repo](https://img.shields.io/badge/GitHub-IDFC-blue?logo=github)](https://github.com/laminendao/idfc)

# 🧠 Interpretable Divisive Feature Clustering (IDFC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides a Python implementation of the **Interpretable Divisive Feature Clustering (IDFC)** algorithm, as introduced in the paper:

> **Explainable Remaining Useful Life Prediction Using an Interpretable Divisive Feature Clustering**  
> M. L. Ndao, G. Youness, N. Niang, G. Saporta – *Submitted 2024*  
> [DOI and paper link coming soon]

---

## 📌 What is IDFC?

**IDFC** is a dimensionality reduction algorithm that creates clusters of highly correlated features in an interpretable and explainable way. It is especially useful in **Explainable AI (XAI)** pipelines for Remaining Useful Life (RUL) prediction, where feature redundancy and multicollinearity limit interpretability.

IDFC combines:
- Hierarchical divisive clustering via **VARCLUS** (for initialization),
- Optimization of internal coherence via **CLV** (Clustering of Variables),
- A **K+1 noise cluster strategy** (Vigneau & Chen, 2016) to isolate atypical variables,
- Final selection of **real, interpretable features**, not latent ones.

---

## ⚙️ Installation

```bash
git clone https://github.com/username/idfc.git
cd idfc
pip install -e .
```

> 📦 A PyPI version is planned following publication.

---

## 📁 Project structure

```
idfc/
├── idfc.py                 # Main pipeline wrapper
├── varclus_init.py         # VARCLUS-based initialization or wrapper
├── clv_refinement.py       # CLV refinement with noise cluster support
├── feature_selection.py    # Selection of representative variables
├── utils.py                # Scoring, correlation, diagnostics
├── notebooks/              # Jupyter examples and visualizations
└── tests/                  # Unit tests
```

---

## 🧪 Quick Example

```python
import pandas as pd
from idfc import run_idfc
from idfc.utils import compute_variable_correlations

X = pd.read_csv("my_data.csv")
selected_features, clusters = run_idfc(X, rho=0.4)

print("Selected interpretable features:", selected_features)

# Show correlations to latent component
correlations = compute_variable_correlations(X, clusters, components=None)
for cluster, df in correlations.items():
    print(f"\nCluster {cluster}")
    print(df)
```

---

## ✅ Features

- No need to predefine the number of clusters (VARCLUS-based)
- Isolates atypical/noise variables (K+1 strategy)
- Returns **interpretable features**, not abstract latent dimensions
- Robust to multicollinearity and redundancy
- Compatible with **SHAP** and other post-hoc XAI tools
- Lightweight, modular and reproducible

---

## 📘 References

- Ndao et al. (2024) – Explainable Remaining Useful Life Prediction Using IDFC *(Preprint pending)*
- Vigneau & Chen (2016) – [Dimensionality Reduction by Clustering of Variables While Setting Aside Atypical Variables](https://doi.org/10.1285/i20705948v9n1p134)
- Sarle (1990) – The VARCLUS Procedure (SAS Institute)

---

## 🧬 Dataset Compatibility

IDFC has been validated on the NASA **C-MAPSS** dataset (prognostics), but is suitable for:
- Multivariate sensor time series
- Genomics and omics data
- Survey/psychometrics
- Any high-dimensional tabular dataset

---

## 📤 Citing this code

If you use this package in a scientific publication, please cite the associated paper (link coming soon) and consider including this GitHub repository URL.

---

## 📜 License

This project is licensed under the MIT License – free to use, modify, and redistribute.