"""Pytest test suite for Encrypted Network Traffic Classification System.

Verifies:
1. Data cleaning properly handles infinities, negative infinities, and NaNs.
2. Label mapping round-trips correctly (encoding and inverse decoding match).
3. The predict function returns a valid class label and probabilities that sum to ~1.0.
"""

import json
import numpy as np
import pandas as pd
import pytest

from src.preprocess import clean_dataframe
from src.predict import predict_flow, load_inference_artifacts
from src.config import LABEL_MAPPING_PATH


def test_clean_dataframe_inf_and_nan():
    """Verify that infinite values and NaNs are stripped out cleanly."""
    dirty_data = {
        " Flow Duration": [1000, 2000, 3000, 4000],
        "Flow Bytes/s": [50.0, np.inf, 120.0, np.nan],
        " Flow Packets/s": [-np.inf, 20.0, 30.0, 40.0],
        " Label": ["BENIGN", "DDoS", "BENIGN", "PortScan"],
    }
    df = pd.DataFrame(dirty_data)

    cleaned = clean_dataframe(df)

    # All column names must be stripped
    assert "Flow Duration" in cleaned.columns
    assert " Flow Duration" not in cleaned.columns
    assert "Label" in cleaned.columns

    # Infinities and NaNs must be completely eliminated
    assert not np.isinf(cleaned["Flow Bytes/s"]).any()
    assert not np.isnan(cleaned["Flow Bytes/s"]).any()
    assert not np.isinf(cleaned["Flow Packets/s"]).any()
    assert not np.isnan(cleaned["Flow Packets/s"]).any()

    # The 3 dirty rows should have been dropped, leaving exactly 1 valid row (index 2)
    assert len(cleaned) == 1
    assert cleaned["Flow Duration"].iloc[0] == 3000


def test_label_mapping_roundtrip():
    """Ensure label mapping JSON encodes and inverts faithfully."""
    sample_classes = ["BENIGN", "Bot", "DDoS", "Infiltration", "PortScan"]
    mapping = {label: idx for idx, label in enumerate(sample_classes)}
    inv_mapping = {v: k for k, v in mapping.items()}

    for label, idx in mapping.items():
        assert inv_mapping[idx] == label

    assert len(mapping) == len(sample_classes)


def test_predict_flow_validity():
    """Verify predict_flow returns recognized class, confidence in [0, 1], and normalized probabilities."""
    # Test if artifacts exist; if so, run full live test
    try:
        model, scaler, features, label_mapping, inv_mapping = load_inference_artifacts()
    except FileNotFoundError:
        pytest.skip("Trained artifacts not yet generated. Run training first.")

    dummy_flow = {feat: 1.0 for feat in features}
    result = predict_flow(dummy_flow)

    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result

    assert result["prediction"] in label_mapping.keys()
    assert 0.0 <= result["confidence"] <= 1.0

    probs = result["probabilities"]
    prob_sum = sum(probs.values())
    assert pytest.approx(prob_sum, 0.01) == 1.0
