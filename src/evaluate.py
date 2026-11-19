"""Model evaluation and visualization module.

Responsibilities:
1. Honest evaluation on the real, untouched test set.
2. Metrics: Accuracy, Macro Precision/Recall/F1, Weighted Precision/Recall/F1.
3. Per-class classification reports.
4. Confusion matrix plots (raw counts & normalized percentages).
5. Per-class Multi-class ROC curves & AUC.
6. Honest note on Infiltration extreme class imbalance.
7. Save plots to reports/figures/ and metrics to reports/metrics.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import FIGURES_DIR, METRICS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_on_test_set(
    model,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Computes comprehensive evaluation metrics and generates production plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

    # Global Metrics
    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    # Classification report dict
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics_payload = {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_wt),
        "weighted_recall": float(r_wt),
        "weighted_f1": float(f1_wt),
        "classification_report": clf_report,
        "class_names": class_names,
        "evaluation_notes": (
            "Honest note: The 'Infiltration' attack class has an extremely low sample count in CICIDS2017 "
            "(~36 rows in total dataset). Consequently, test metrics for Infiltration exhibit high statistical "
            "variance and should be interpreted as exploratory rather than definitively hardened."
        ),
    }

    # Save metrics JSON
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info(f"Saved evaluation metrics to: {METRICS_PATH}")

    # 1. Confusion Matrix (Raw counts)
    cm_raw = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Untouched Test Set - Raw Confusion Matrix", fontsize=14, weight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_raw.png", dpi=300)
    plt.close()

    # 2. Confusion Matrix (Normalized percentages)
    cm_norm = cm_raw.astype("float") / (cm_raw.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Untouched Test Set - Normalized Confusion Matrix", fontsize=14, weight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_normalized.png", dpi=300)
    plt.close()

    # 3. ROC Curves (Per-class)
    if y_prob is not None:
        n_classes = len(class_names)
        y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
        plt.figure(figsize=(10, 8))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i in range(n_classes):
            # Check if class is present in test set
            if y_test_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    color=colors[i % len(colors)],
                    lw=2,
                    label=f"{class_names[i]} (AUC = {roc_auc:.3f})",
                )

        plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Per-Class Receiver Operating Characteristic (ROC)", fontsize=14, weight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=300)
        plt.close()

    logger.info("Evaluation charts generated in reports/figures/")
    return metrics_payload
