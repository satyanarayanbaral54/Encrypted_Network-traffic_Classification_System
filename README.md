# Encrypted Network Traffic Classification System

A production-grade, leak-free machine learning system designed to classify encrypted network flows and detect cyber threats using the **CICIDS2017** benchmark dataset.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 🌟 Key Highlights

1. **Zero Data Leakage Pipeline**:
   - Stratified train/test split (80/20) executed **first**.
   - The test set (`data/processed/test_set.csv`) is preserved untouched.
   - Feature selection (top 20 via Random Forest Gini importance) and `StandardScaler` are fitted exclusively on the training partition.
2. **Multi-Model Benchmark & Tuning**:
   - Baselines evaluated: Logistic Regression, Decision Tree, K-Nearest Neighbors, Random Forest, and XGBoost.
   - Fast hyperparameter tuning via `GridSearchCV` on a 50K stratified subsample, then refitted on the full training split.
3. **Honest Evaluation**:
   - Confusion matrices (raw & normalized), multi-class ROC-AUC curves, and per-class classification reports.
   - Transparent reporting on extreme class imbalance (e.g. Infiltration attack rarity).
4. **Interactive Streamlit Dashboard**:
   - 8 comprehensive operational pages: Home, Data Explorer, Single Flow Predictor, Batch Processor, Live Wire Simulator, Performance Benchmarks, SHAP Explainability, and System Architecture.

---

## 📁 Project Structure

```text
network-traffic-classifier/
├── app/
│   └── streamlit_app.py          # 8-page interactive Streamlit dashboard
├── data/
│   ├── raw/                      # 4 raw CICIDS2017 CSV captures
│   └── processed/                # Untouched test_set.csv
├── models/
│   ├── best_model.pkl            # Final production classifier
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── selected_features.pkl     # Top 20 feature names
│   ├── label_mapping.json        # String-to-integer class mappings
│   └── metadata.json             # Model training metadata & parameters
├── reports/
│   ├── figures/                  # Confusion matrix, ROC curves, feature importance
│   ├── model_comparison.csv      # Baseline vs tuned benchmark table
│   └── metrics.json              # Final honest test metrics
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, random seeds, and global constants
│   ├── preprocess.py             # Data cleaning, deduplication, stratified split
│   ├── features.py               # Feature selection, scaling, SMOTE vs class balancing
│   ├── train.py                  # Multi-model training and grid hyperparameter tuning
│   ├── evaluate.py               # Honest test set evaluation & visualization
│   └── predict.py                # Reusable single-flow and batch inference engine
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # Pytest verification suite
├── requirements.txt              # Pinned dependencies
└── README.md                     # Documentation
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Entire End-to-End ML Pipeline
Executes data preprocessing, leak-free feature selection, model training, hyperparameter tuning, and test set evaluation in one command:
```bash
python -m src.train
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 4. Run the Unit Test Suite
```bash
pytest tests/ -v
```

---

## 📊 Dataset & Class Distribution

The dataset comprises 4 primary attack sessions from CICIDS2017:
- **Friday Afternoon DDoS**: High-volume UDP/TCP volumetric floods.
- **Friday Afternoon PortScan**: Rapid reconnaissance probing host TCP/UDP ports.
- **Friday Morning Bot**: Zeus and Ares botnet command-and-control telemetry.
- **Thursday Afternoon Infiltration**: Internal network reconnaissance and vulnerability exploitation.

| Class Label | Description | Operational Significance |
| :--- | :--- | :--- |
| **BENIGN** | Normal day-to-day corporate network traffic | Baseline normal activity |
| **DDoS** | Volumetric socket saturation and request floods | Service availability attack |
| **PortScan** | Network perimeter host and port discovery | Reconnaissance phase |
| **Bot** | Automated agent communication with C2 servers | Persistent threat activity |
| **Infiltration** | Deep network exploitation & pivot attempts | High-risk targeted intrusion |

> [!IMPORTANT]
> **Notice Regarding Infiltration Class**:
> The Infiltration class contains only ~36 total records across the entire dataset. While stratified splitting ensures test representation, metrics for this specific attack class are subject to wide confidence intervals and must be interpreted as exploratory.

---

## 📈 Benchmark & Evaluation Summary

| Metric | Test Split Value |
| :--- | :--- |
| **Accuracy** | > 99.8% |
| **Macro F1-Score** | > 0.96 |
| **Weighted F1-Score** | > 0.99 |
| **ROC AUC (Multi-Class)** | > 0.99 |

*Detailed confusion matrices, per-class metrics, and ROC charts are automatically exported to `reports/figures/` and `reports/metrics.json`.*
