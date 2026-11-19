"""Flask API server for the Network Traffic Classification System.

Serves the static frontend and provides REST API endpoints for:
- Model metadata & metrics
- Single flow prediction
- Batch CSV prediction
- Test set sample data for explorer & simulation
- Report images (confusion matrix, ROC, feature importance)
"""

import base64
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Paths
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TEST_SET_PATH = BASE_DIR / "data" / "processed" / "test_set.csv"

# Cache for loaded artifacts
_cache = {}


def load_artifacts():
    """Load and cache model, scaler, features, and label mapping."""
    if "model" not in _cache:
        best_model_path = MODELS_DIR / "best_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        features_path = MODELS_DIR / "selected_features.pkl"
        label_path = MODELS_DIR / "label_mapping.json"
        metadata_path = MODELS_DIR / "metadata.json"

        if not best_model_path.exists():
            return False

        _cache["model"] = joblib.load(best_model_path)
        _cache["scaler"] = joblib.load(scaler_path)
        _cache["features"] = joblib.load(features_path)

        with open(label_path, "r") as f:
            mapping = json.load(f)
        _cache["label_mapping"] = mapping
        _cache["inv_mapping"] = {v: k for k, v in mapping.items()}

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        _cache["metadata"] = metadata

    return True


# ── Static file serving ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ── API: Metadata ────────────────────────────────────────────────────────────

@app.route("/api/metadata")
def get_metadata():
    load_artifacts()
    metadata = _cache.get("metadata", {})
    features = _cache.get("features", [])
    inv_mapping = _cache.get("inv_mapping", {})
    return jsonify({
        "metadata": metadata,
        "features": list(features) if features else [],
        "classes": list(inv_mapping.values()) if inv_mapping else [],
        "model_loaded": "model" in _cache,
    })


@app.route("/api/metrics")
def get_metrics():
    metrics_file = REPORTS_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found"}), 404


@app.route("/api/model-comparison")
def get_model_comparison():
    comp_file = REPORTS_DIR / "model_comparison.csv"
    if comp_file.exists():
        df = pd.read_csv(comp_file)
        return jsonify(df.to_dict(orient="records"))
    return jsonify({"error": "Comparison file not found"}), 404


# ── API: Figures ─────────────────────────────────────────────────────────────

@app.route("/api/figure/<name>")
def get_figure(name):
    allowed = {
        "confusion_matrix_normalized": "confusion_matrix_normalized.png",
        "confusion_matrix_raw": "confusion_matrix_raw.png",
        "roc_curves": "roc_curves.png",
        "feature_importance": "feature_importance.png",
    }
    filename = allowed.get(name)
    if filename:
        path = FIGURES_DIR / filename
        if path.exists():
            return send_file(str(path), mimetype="image/png")
    return jsonify({"error": "Figure not found"}), 404


# ── API: Test data samples ──────────────────────────────────────────────────

@app.route("/api/test-sample")
def get_test_sample():
    n = request.args.get("n", 200, type=int)
    n = min(n, 2000)
    if TEST_SET_PATH.exists():
        df = pd.read_csv(TEST_SET_PATH, nrows=n)
        # Replace inf/nan for JSON safety
        df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return jsonify({
            "columns": df.columns.tolist(),
            "data": df.head(50).to_dict(orient="records"),
            "total_rows": len(df),
            "class_distribution": df["Label"].value_counts().to_dict() if "Label" in df.columns else {},
        })
    return jsonify({"error": "Test set not found"}), 404


@app.route("/api/feature-stats")
def get_feature_stats():
    """Returns statistical summary for all numeric features."""
    if TEST_SET_PATH.exists():
        df = pd.read_csv(TEST_SET_PATH, nrows=1000)
        df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        features = _cache.get("features", [])
        available = [f for f in features if f in df.columns]
        if available:
            stats = df[available].describe().to_dict()
            corr = df[available].corr().replace([np.inf, -np.inf], 0.0).fillna(0.0).to_dict()
            return jsonify({"stats": stats, "correlation": corr, "features": available})
    return jsonify({"error": "Data not available"}), 404


# ── API: Single prediction ──────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict_single():
    if not load_artifacts():
        return jsonify({"error": "Model not loaded"}), 500

    model = _cache["model"]
    scaler = _cache["scaler"]
    features = _cache["features"]
    inv_mapping = _cache["inv_mapping"]

    data = request.json
    flow_dict = data.get("flow", {})

    row_data = {}
    for feat in features:
        val = flow_dict.get(feat, 0.0)
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
            probs[cls_name] = round(float(p), 6)
        confidence = round(float(np.max(prob_arr)), 6)

    # Feature attributions (importance * scaled value)
    if hasattr(model, "feature_importances_"):
        tree_importances = model.feature_importances_
    else:
        tree_importances = np.ones(len(features)) / len(features)

    scaled_vec = X_scaled[0]
    attributions = (scaled_vec * tree_importances).tolist()

    return jsonify({
        "prediction": pred_class,
        "confidence": confidence,
        "probabilities": probs,
        "attributions": dict(zip(features, [round(a, 6) for a in attributions])),
    })


# ── API: Preset flow samples ────────────────────────────────────────────────

@app.route("/api/preset/<preset_type>")
def get_preset(preset_type):
    if not load_artifacts():
        return jsonify({"error": "Model not loaded"}), 500

    features = _cache["features"]
    if not TEST_SET_PATH.exists():
        return jsonify({"error": "Test set not found"}), 404

    df = pd.read_csv(TEST_SET_PATH, nrows=500)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if preset_type == "random":
        row = df.sample(1).iloc[0]
    else:
        label_col = "Label" if "Label" in df.columns else df.columns[-1]
        matches = df[df[label_col].astype(str).str.contains(preset_type, case=False, na=False)]
        if len(matches) > 0:
            row = matches.sample(1).iloc[0]
        else:
            row = df.sample(1).iloc[0]

    flow_data = {}
    for f in features:
        flow_data[f] = float(row.get(f, 0.0))

    return jsonify({"flow": flow_data})


# ── API: Batch prediction ──────────────────────────────────────────────────

@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    if not load_artifacts():
        return jsonify({"error": "Model not loaded"}), 500

    model = _cache["model"]
    scaler = _cache["scaler"]
    features = _cache["features"]
    inv_mapping = _cache["inv_mapping"]

    # Handle file upload or sample request
    if "file" in request.files:
        file = request.files["file"]
        df = pd.read_csv(file)
    elif request.json and request.json.get("use_sample"):
        n = request.json.get("sample_size", 200)
        if TEST_SET_PATH.exists():
            df = pd.read_csv(TEST_SET_PATH, nrows=n)
        else:
            return jsonify({"error": "Test set not found"}), 404
    else:
        return jsonify({"error": "No data provided"}), 400

    # Clean
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    X = pd.DataFrame(index=df_clean.index)
    for feat in features:
        if feat in df_clean.columns:
            col = pd.to_numeric(df_clean[feat], errors="coerce").fillna(0.0)
            col = col.replace([np.inf, -np.inf], 0.0)
            X[feat] = col
        else:
            X[feat] = 0.0

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    pred_labels = [inv_mapping.get(int(idx), f"Class_{idx}") for idx in preds]

    confidences = []
    if hasattr(model, "predict_proba"):
        prob_matrix = model.predict_proba(X_scaled)
        confidences = [round(float(p), 4) for p in np.max(prob_matrix, axis=1)]
    else:
        confidences = [1.0] * len(preds)

    results = df.copy()
    results = results.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    results["Predicted_Class"] = pred_labels
    results["Confidence"] = confidences

    class_counts = pd.Series(pred_labels).value_counts().to_dict()

    return jsonify({
        "total_flows": len(results),
        "class_distribution": class_counts,
        "results": results.head(100).to_dict(orient="records"),
        "columns": results.columns.tolist(),
    })


# ── API: Batch predict from sample for live simulation ──────────────────────

@app.route("/api/simulate-batch")
def simulate_batch():
    """Return a batch of pre-predicted flows for the live simulation page."""
    if not load_artifacts():
        return jsonify({"error": "Model not loaded"}), 500

    model = _cache["model"]
    scaler = _cache["scaler"]
    features = _cache["features"]
    inv_mapping = _cache["inv_mapping"]

    n = request.args.get("n", 50, type=int)
    n = min(n, 200)

    if not TEST_SET_PATH.exists():
        return jsonify({"error": "Test set not found"}), 404

    df = pd.read_csv(TEST_SET_PATH, nrows=500)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    sampled = df.sample(n, replace=True).reset_index(drop=True)

    X = pd.DataFrame(index=sampled.index)
    for feat in features:
        if feat in sampled.columns:
            X[feat] = pd.to_numeric(sampled[feat], errors="coerce").fillna(0.0)
        else:
            X[feat] = 0.0

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    flows = []
    if hasattr(model, "predict_proba"):
        prob_matrix = model.predict_proba(X_scaled)

    for i in range(len(preds)):
        pred_class = inv_mapping.get(int(preds[i]), f"Class_{preds[i]}")
        conf = float(np.max(prob_matrix[i])) if hasattr(model, "predict_proba") else 1.0
        feature_vals = {f: round(float(sampled.iloc[i].get(f, 0.0)), 4) for f in features[:5]}
        flows.append({
            "id": i + 1,
            "prediction": pred_class,
            "confidence": round(conf, 4),
            "is_attack": "BENIGN" not in pred_class.upper(),
            "features_preview": feature_vals,
        })

    return jsonify({"flows": flows})


if __name__ == "__main__":
    print("=" * 60)
    print("  Network Traffic Classification - Web Dashboard")
    print("  Open in browser: http://localhost:5000")
    print("=" * 60)
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=True)
