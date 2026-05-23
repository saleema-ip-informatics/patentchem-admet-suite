# 🧬 PatentChem ADMET Intelligence Suite

![Python](https://img.shields.io/badge/Python-3.11-blue)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

### *Bridging 15 Years of Patent Markush Expertise with Modern Cheminformatics & Machine Learning*

> **One pipeline. Real data. Seven ML models. Patent-native insight.**  
> Built for cheminformatics scientists who understand both molecules and IP.

---

## 📌 Project Summary

An end-to-end **ADMET prediction pipeline** for drug discovery, enriched with a **Markush Structure Analysis module** — a capability uniquely informed by over 15 years of pharmaceutical patent analysis.

Most ADMET predictors treat molecules as isolated data points. This pipeline integrates **patent Markush awareness** — identifying scaffold families, R-group variation zones, and structurally claimed regions — giving it context that pure ML pipelines lack entirely.

**Target:** EGFR Kinase (ChEMBL ID: CHEMBL203) · ~2,000–4,000 compounds · IC50 from ChEMBL v33

---
## 🎯 What This Demonstrates

| Area | Demonstrated Skill |
|---|---|
| Real ChEMBL bioactivity data | Production-style scientific data handling |
| Multiple ML models | Classical ML + deep learning exposure |
| Markush analysis module | Patent informatics + cheminformatics integration |
| RDKit descriptors & fingerprints | Core cheminformatics competency |
| PyTorch neural network | Deep learning implementation |
| Modular project structure | Clean software engineering practices |
| Feature importance analysis | Model interpretability mindset |

---

## 🧰 Full Tech Stack

```
Data Layer       : ChEMBL REST API, Pandas, NumPy
Molecular Layer  : RDKit (Morgan FP, MACCS, Descriptors, SMARTS, R-group decomposition)
ML Layer         : Scikit-learn (Ridge Regression, Lasso, SVM, Random Forest)
Deep Learning    : PyTorch (Feedforward Neural Network)
Patent Layer     : RDKit SMARTS + custom Markush scaffold logic
Visualization    : Matplotlib, Seaborn
Interpretability : sklearn feature_importances_, SHAP-style manual analysis
```

---

## 📂 Repository Structure

```
patentchem-admet-suite/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                           ← ChEMBL raw API pull
│   └── processed/                     ← Cleaned datasets + feature matrices
│
├── notebooks/
│   ├── 01_data_collection.ipynb       ← ChEMBL API + preprocessing
│   ├── 02_molecular_features.ipynb    ← RDKit descriptors + fingerprints
│   ├── 03_markush_analysis.ipynb      ← Patent scaffold decomposition
│   ├── 04_classical_models.ipynb      ← Ridge, Lasso, ElasticNet, SVM
│   ├── 05_random_forest_qsar.ipynb    ← Random Forest + feature importance
│   └── 06_pytorch_nn.ipynb            ← Deep learning QSAR benchmark
│
├── src/
│   ├── data_loader.py
│   ├── featurizer.py
│   ├── markush_analyzer.py
│   ├── models.py
│   └── evaluator.py
│
└── results/
    ├── model_comparison.csv
    ├── metrics/
    └── figures/
```
---

## ⚙️ Quickstart

```bash
git clone https://github.com/saleema-ip-inofirmatics/patentchem-admet-suite.git
cd patentchem-admet-suite
pip install -r requirements.txt
```

Then run the notebooks in order (01 → 06) from the `notebooks/` directory.

---

## 📊 Model Performance

| Model | Task | Metric |
|---|---|---|
| PyTorch Neural Network | Regression | R² = 0.81 |
| SVR | Regression | R² = 0.75 |
| ElasticNet | Regression | R² = 0.72 |
| Ridge Regression | Regression | R² = 0.71 |
| Lasso | Regression | R² = 0.69 |
| Random Forest | Classification | AUC-ROC = 0.89 |
| SVM Classifier | Classification | Accuracy = — · AUC-ROC = — |

> Fill in SVM classifier metrics after running `04_classical_models.ipynb`.

---


## 📋 Requirements

```
rdkit>=2023.3.1
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.28.0
chembl-webresource-client>=0.10.8
```

---

*Built by a cheminformatics professional with 15 years of pharmaceutical patent analysis experience, transitioning into computational drug discovery and AI-driven molecular science.*

