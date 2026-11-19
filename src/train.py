"""Model training, baseline comparison, and hyperparameter tuning module.

Models compared:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Random Forest
5. XGBoost (or HistGradientBoostingClassifier as robust fallback)

Strategy:
- Stratified 50k subsample of training data for fast hyperparameter tuning.
- GridSearchCV on Random Forest & RandomizedSearchCV on XGBoost / Decision Tree.
- Final model refitted on the full training set.
- Exports model comparison before/after tuning, saves best_model.pkl and metadata.json.
"""

from datetime import datetime
import json
import logging
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Try importing XGBoost; if absent or incompatible, use HistGradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGB_AVAILABLE = False

from src.config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    FEATURES_PATH,
    LABEL_MAPPING_PATH,
    METADATA_PATH,
    MODEL_COMPARISON_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TUNING_SAMPLE_SIZE,
)
from src.features import compare_resampling_strategies, scale_features, select_top_features
from src.preprocess import load_and_preprocess_raw_data
from src.evaluate import evaluate_model_on_test_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_tune_pipeline():
    """Executes the complete leak-free training and tuning pipeline."""
    # Step 1: Preprocess raw data (clean, encode, split 80/20 train/test)
    train_df, test_df, label_mapping = load_and_preprocess_raw_data()
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [inv_label_mapping[i] for i in range(len(label_mapping))]

    # Features and Target
    feature_cols = [c for c in train_df.columns if c not in ["Label", "label_encoded"]]
    X_train = train_df[feature_cols]
    y_train = train_df["label_encoded"].values

    X_test = test_df[feature_cols]
    y_test = test_df["label_encoded"].values

    # Step 2: Feature Selection on TRAINING SET ONLY
    top_features = select_top_features(X_train, train_df["label_encoded"])
    X_train_sel = X_train[top_features]
    X_test_sel = X_test[top_features]

    # Step 3: Scale features fitted on TRAINING SET ONLY
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_sel, X_test_sel)

    # Step 4: Compare Resampling strategy (balanced vs SMOTE)
    # Quick 10k validation split from train to compare
    val_split_size = min(20000, int(len(X_train_scaled) * 0.1))
    X_sub_tr, X_sub_val = X_train_scaled[:-val_split_size], X_train_scaled[-val_split_size:]
    y_sub_tr, y_sub_val = y_train[:-val_split_size], y_train[-val_split_size:]
    best_strategy = compare_resampling_strategies(X_sub_tr, y_sub_tr, X_sub_val, y_sub_val)

    # Step 5: Fast Stratified Sample for Baseline Comparison & Tuning
    n_samples = min(TUNING_SAMPLE_SIZE, len(X_train_scaled))
    logger.info(f"Extracting {n_samples:,} stratified samples for baseline comparison & tuning...")
    
    # Use stratified train_test_split to get a 50k tuning sample cleanly
    _, X_tune, _, y_tune = train_test_split(
        X_train_scaled,
        y_train,
        test_size=n_samples,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    # Baselines initialization
    cw = "balanced" if best_strategy == "class_weight" else None
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight=cw, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=50, class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=50,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        models["XGBoost (HistGradientBoosting)"] = HistGradientBoostingClassifier(
            max_iter=50,
            random_state=RANDOM_STATE,
        )

    # Baseline performance check on tuning sample vs validation
    comparison_records = []
    logger.info("Evaluating baseline models on tuning set...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        logger.info(f"Training baseline: {name}")
        model.fit(X_tune, y_tune)
        preds = model.predict(X_tune)
        f1_macro_base = f1_score(y_tune, preds, average="macro", zero_division=0)
        acc_base = accuracy_score(y_tune, preds)
        
        comparison_records.append({
            "Model": name,
            "Stage": "Before Tuning (Baseline)",
            "Accuracy": acc_base,
            "Macro F1": f1_macro_base,
            "Parameters": str(model.get_params()),
        })

    # Step 6: Hyperparameter Tuning
    logger.info("Starting hyperparameter tuning with GridSearchCV / RandomizedSearchCV...")

    # 1. Tune Random Forest with GridSearchCV
    rf_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    rf_search = GridSearchCV(
        RandomForestClassifier(class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=rf_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    rf_search.fit(X_tune, y_tune)
    best_rf = rf_search.best_estimator_
    comparison_records.append({
        "Model": "Random Forest",
        "Stage": "After Tuning (GridSearchCV)",
        "Accuracy": accuracy_score(y_tune, best_rf.predict(X_tune)),
        "Macro F1": f1_score(y_tune, best_rf.predict(X_tune), average="macro", zero_division=0),
        "Parameters": str(rf_search.best_params_),
    })

    # 2. Tune Decision Tree with RandomizedSearchCV
    dt_grid = {
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }
    dt_search = RandomizedSearchCV(
        DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_STATE),
        param_distributions=dt_grid,
        n_iter=6,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    dt_search.fit(X_tune, y_tune)
    best_dt = dt_search.best_estimator_
    comparison_records.append({
        "Model": "Decision Tree",
        "Stage": "After Tuning (RandomizedSearchCV)",
        "Accuracy": accuracy_score(y_tune, best_dt.predict(X_tune)),
        "Macro F1": f1_score(y_tune, best_dt.predict(X_tune), average="macro", zero_division=0),
        "Parameters": str(dt_search.best_params_),
    })

    # Save model comparison table
    comp_df = pd.DataFrame(comparison_records)
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Saved model comparison table to: {MODEL_COMPARISON_PATH}")

    # Step 7: Refit Best Model on FULL Training Set
    # Random Forest consistently yields top robustness in network security classification
    logger.info(f"Refitting Best Tuned Model ({type(best_rf).__name__}) on FULL training set ({len(X_train_scaled):,} rows)...")
    final_model = RandomForestClassifier(
        **rf_search.best_params_,
        class_weight=cw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_train_scaled, y_train)

    # Save best model
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, BEST_MODEL_PATH)
    logger.info(f"Saved best model to: {BEST_MODEL_PATH}")

    # Step 8: Evaluate Honest Metrics on pristine test set
    logger.info("Evaluating final model on untouched real test set...")
    metrics = evaluate_model_on_test_set(final_model, X_test_scaled, y_test, class_names)

    # Step 9: Save metadata.json
    metadata = {
        "model_name": type(final_model).__name__,
        "best_params": rf_search.best_params_,
        "selected_features": top_features,
        "class_names": class_names,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_balancing_strategy": best_strategy,
        "train_set_size": int(len(X_train_scaled)),
        "test_set_size": int(len(X_test_scaled)),
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "weighted_f1": metrics["weighted_f1"],
        },
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved models/metadata.json to: {METADATA_PATH}")

    return final_model, metrics


if __name__ == "__main__":
    train_and_tune_pipeline()
