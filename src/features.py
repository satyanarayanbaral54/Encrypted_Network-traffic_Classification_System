"""Feature engineering, feature selection, scaling, and class balancing module.

Responsibilities:
1. Select top 20 features using Random Forest importance computed on the TRAINING set only.
2. Generate feature importance chart (reports/figures/feature_importance.png).
3. Save selected features list (models/selected_features.pkl).
4. Fit StandardScaler on TRAINING set only, transform train & test, save models/scaler.pkl.
5. Compare SMOTE (k_neighbors=3) vs class_weight='balanced' on the training set only.
6. Provide simple, student-friendly explanations of every step.
"""

import logging
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES_PATH,
    FIGURES_DIR,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_COLUMN,
    TOP_K_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = TOP_K_FEATURES,
) -> List[str]:
    """Fits a Random Forest on the TRAINING set only to determine Gini importance,
    extracts the top_k features, saves the feature list and creates an importance plot.

    Why Training set only?
    Evaluating or picking features on the test set causes data leakage, artificially
    inflating performance metrics and hiding model overfitting.
    """
    logger.info(f"Selecting top {top_k} features using Random Forest on TRAINING set...")

    # Fast RF fit on a stratified subsample if X_train is very large (e.g. > 100k rows)
    # to maintain high statistical accuracy while saving runtime.
    sample_size = min(len(X_train), 100000)
    if len(X_train) > sample_size:
        logger.info(f"Subsampling {sample_size:,} stratified rows for feature importance calculation...")
        X_sub = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
        y_sub = y_train.loc[X_sub.index]
    else:
        X_sub, y_sub = X_train, y_train

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_sub, y_sub)

    importances = rf.feature_importances_
    feature_names = X_train.columns.tolist()
    feat_imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_imp_df = feat_imp_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    top_features = feat_imp_df.head(top_k)["feature"].tolist()
    logger.info(f"Selected top {top_k} features: {top_features}")

    # Save selected features list to models/selected_features.pkl
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(top_features, FEATURES_PATH)
    logger.info(f"Saved selected features list to: {FEATURES_PATH}")

    # Save feature importance chart
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=feat_imp_df.head(top_k),
        x="importance",
        y="feature",
        palette="viridis",
        hue="feature",
        legend=False,
    )
    plt.title(f"Top {top_k} Features by Random Forest Importance (Training Split Only)", fontsize=14, weight="bold")
    plt.xlabel("Mean Impurity Decrease (Gini Importance)", fontsize=12)
    plt.ylabel("Flow Feature", fontsize=12)
    plt.tight_layout()
    chart_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance chart to: {chart_path}")

    return top_features


def scale_features(
    X_train_selected: pd.DataFrame,
    X_test_selected: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on the TRAINING set only, then transform both train & test.

    Why?
    Fitting the scaler on test data leaks the test set's mean and variance into the model.
    Fitting only on train ensures strict leak-free inference behavior.
    """
    logger.info("Fitting StandardScaler on training set only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Persist fitted scaler
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Saved fitted scaler to: {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, scaler


def compare_resampling_strategies(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
) -> str:
    """Compares SMOTE (k_neighbors=3) vs class_weight='balanced' on the training set.
    
    Infiltration has very few samples in CICIDS2017, so SMOTE requires a small k_neighbors (k=3).
    We test a quick Random Forest on a validation split to see which yields a higher macro F1.
    """
    logger.info("Comparing SMOTE (k_neighbors=3) vs class_weight='balanced' on training data...")

    # Strategy 1: class_weight='balanced'
    rf_balanced = RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_balanced.fit(X_train_scaled, y_train)
    preds_balanced = rf_balanced.predict(X_val_scaled)
    f1_balanced = f1_score(y_val, preds_balanced, average="macro")
    logger.info(f"Strategy class_weight='balanced' Validation Macro F1: {f1_balanced:.4f}")

    # Strategy 2: SMOTE (k_neighbors=3)
    # Check minimum class count in y_train
    class_counts = pd.Series(y_train).value_counts()
    min_count = class_counts.min()
    logger.info(f"Smallest class count in train split: {min_count}")

    f1_smote = -1.0
    if min_count > 3:
        try:
            smote = SMOTE(k_neighbors=3, random_state=RANDOM_STATE)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            rf_smote = RandomForestClassifier(
                n_estimators=50,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            rf_smote.fit(X_train_smote, y_train_smote)
            preds_smote = rf_smote.predict(X_val_scaled)
            f1_smote = f1_score(y_val, preds_smote, average="macro")
            logger.info(f"Strategy SMOTE (k=3) Validation Macro F1: {f1_smote:.4f}")
        except Exception as e:
            logger.warning(f"SMOTE application failed: {e}. Defaulting to class_weight='balanced'.")
            f1_smote = -1.0
    else:
        logger.info(f"Smallest class count ({min_count}) is <= 3, cannot reliably apply SMOTE.")

    if f1_smote > f1_balanced:
        chosen = "smote"
        logger.info(f"Chosen strategy: SMOTE (k=3) with Macro F1: {f1_smote:.4f}")
    else:
        chosen = "class_weight"
        logger.info(f"Chosen strategy: class_weight='balanced' with Macro F1: {f1_balanced:.4f}")

    return chosen
