"""Reusable prediction function and inference engine for flows and batches.

Handles:
1. Feature extraction matching training set.
2. Handling inf/NaNs gracefully.
3. Scaling using saved scaler.pkl.
4. Returning predicted class label and calibrated probabilities.
"""

import json
import logging
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, FEATURES_PATH, LABEL_MAPPING_PATH, SCALER_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_MODEL_CACHE = {}


def load_inference_artifacts():
    """Loads and caches model, scaler, features list, and label mapping."""
    if "model" not in _MODEL_CACHE:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {BEST_MODEL_PATH}. Run training first.")
        _MODEL_CACHE["model"] = joblib.load(BEST_MODEL_PATH)
        _MODEL_CACHE["scaler"] = joblib.load(SCALER_PATH)
        _MODEL_CACHE["features"] = joblib.load(FEATURES_PATH)
        with open(LABEL_MAPPING_PATH, "r") as f:
            mapping = json.load(f)
        _MODEL_CACHE["label_mapping"] = mapping
        _MODEL_CACHE["inv_mapping"] = {v: k for k, v in mapping.items()}

    return (
        _MODEL_CACHE["model"],
        _MODEL_CACHE["scaler"],
        _MODEL_CACHE["features"],
        _MODEL_CACHE["label_mapping"],
        _MODEL_CACHE["inv_mapping"],
    )


def predict_flow(flow_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predicts traffic class for a single network flow dictionary.

    Args:
        flow_dict: Key-value dictionary containing flow metrics.

    Returns:
        Dict with 'prediction', 'probabilities' per class, and 'confidence'.
    """
    model, scaler, selected_features, _, inv_mapping = load_inference_artifacts()

    # Build 1-row DataFrame, filling missing features with 0.0
    row_data = {}
    for feat in selected_features:
        # Check direct or stripped match
        val = flow_dict.get(feat, flow_dict.get(feat.strip(), 0.0))
        try:
            val = float(val)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
        except (ValueError, TypeError):
            val = 0.0
        row_data[feat] = val

    df = pd.DataFrame([row_data])
    X_scaled = scaler.transform(df)

    pred_idx = int(model.predict(X_scaled)[0])
    pred_class = inv_mapping.get(pred_idx, f"Class_{pred_idx}")

    probs = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(X_scaled)[0]
        for idx, p in enumerate(prob_arr):
            cls_name = inv_mapping.get(idx, f"Class_{idx}")
            probs[cls_name] = float(p)
        confidence = float(np.max(prob_arr))

    return {
        "prediction": pred_class,
        "confidence": confidence,
        "probabilities": probs,
    }


def predict_batch(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Runs batch inference on an input DataFrame.
    
    Args:
        df: Input DataFrame containing flow records.
        
    Returns:
        results_df: Original df + 'Predicted_Class' + 'Confidence'
        class_summary: Count of predictions per class
    """
    model, scaler, selected_features, _, inv_mapping = load_inference_artifacts()

    # Clean columns
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # Align features
    X = pd.DataFrame(index=df_clean.index)
    for feat in selected_features:
        if feat in df_clean.columns:
            col_series = pd.to_numeric(df_clean[feat], errors="coerce").fillna(0.0)
            col_series = col_series.replace([np.inf, -np.inf], 0.0)
            X[feat] = col_series
        else:
            X[feat] = 0.0

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    pred_classes = [inv_mapping.get(int(idx), f"Class_{idx}") for idx in preds]

    confidences = []
    if hasattr(model, "predict_proba"):
        prob_matrix = model.predict_proba(X_scaled)
        confidences = [float(p) for p in np.max(prob_matrix, axis=1)]
    else:
        confidences = [1.0] * len(preds)

    results_df = df.copy()
    results_df["Predicted_Class"] = pred_classes
    results_df["Confidence"] = confidences

    class_counts = results_df["Predicted_Class"].value_counts().to_dict()
    return results_df, class_counts
