"""Data preprocessing module for Encrypted Network Traffic Classification System.

Responsibilities:
1. Load 4 target raw CSVs from data/raw/
2. Strip leading/trailing whitespaces from column headers.
3. Clean invalid numbers: convert inf / -inf to NaN, drop NaN rows, drop constant columns.
4. Encode labels with LabelEncoder and save label mapping JSON.
5. Perform STRATIFIED 80/20 train/test split.
6. Persist untouched test set to data/processed/test_set.csv.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    LABEL_MAPPING_PATH,
    RANDOM_STATE,
    RAW_CSV_FILES,
    RAW_DATA_DIR,
    TARGET_COLUMN,
    TEST_SET_PATH,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, handle inf/NaN, and drop duplicate rows.
    
    Args:
        df: Input raw or dirty dataframe.
        
    Returns:
        Cleaned dataframe without infinities or missing records.
    """
    # Step 1: Strip leading and trailing whitespace from column names
    # Example: " Label" -> "Label"
    df.columns = [c.strip() for c in df.columns]

    # Step 2: Drop duplicated rows to prevent redundant traffic inflation
    initial_len = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {initial_len - len(df):,} duplicated rows.")

    # Step 3: Replace inf, -inf with NaN for standard numeric cleanup
    # In network datasets, Flow Bytes/s and Flow Packets/s often contain division-by-zero infs.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Step 4: Drop any remaining NaN rows
    before_na = len(df)
    df = df.dropna()
    logger.info(f"Dropped {before_na - len(df):,} rows containing NaN or infinite values.")

    return df


def load_and_preprocess_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Loads 4 raw CICIDS2017 CSV files, cleans them, applies stratified split,
    and saves the untouched test set along with label mapping.
    
    Returns:
        train_df: Cleaned training set (80%)
        test_df: Pristine test set (20%)
        label_mapping: Dict mapping original string label to integer encoded value
    """
    dfs = []
    for filename in RAW_CSV_FILES:
        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Expected raw file not found at: {filepath}")
        logger.info(f"Loading {filename}...")
        sub_df = pd.read_csv(filepath, low_memory=False)
        dfs.append(sub_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined raw shape: {combined_df.shape}")

    # Clean the dataset
    cleaned_df = clean_dataframe(combined_df)

    # Standardize labels: strip spaces from class labels
    # E.g. "Web Attack  Sql Injection" or " Infiltration " -> "Infiltration"
    cleaned_df[TARGET_COLUMN] = cleaned_df[TARGET_COLUMN].astype(str).str.strip()

    # Drop constant columns (zero variance)
    # Identifying numeric columns with standard deviation 0
    numeric_features = [c for c in cleaned_df.columns if c != TARGET_COLUMN and np.issubdtype(cleaned_df[c].dtype, np.number)]
    constant_cols = [c for c in numeric_features if cleaned_df[c].std() == 0 or cleaned_df[c].nunique() <= 1]
    if constant_cols:
        logger.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        cleaned_df = cleaned_df.drop(columns=constant_cols)

    logger.info(f"Class distribution after cleaning:\n{cleaned_df[TARGET_COLUMN].value_counts()}")

    # Fit LabelEncoder
    le = LabelEncoder()
    cleaned_df["label_encoded"] = le.fit_transform(cleaned_df[TARGET_COLUMN])
    label_mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}

    # Save label mapping to JSON
    LABEL_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAPPING_PATH, "w") as f:
        json.dump(label_mapping, f, indent=4)
    logger.info(f"Saved label mapping to {LABEL_MAPPING_PATH}: {label_mapping}")

    # Stratified Train/Test split FIRST to prevent data leakage
    # Infiltration has few samples, stratified split ensures representation in both splits
    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=cleaned_df["label_encoded"],
    )

    logger.info(f"Train split size: {len(train_df):,}, Test split size: {len(test_df):,}")

    # Save untouched pristine test set
    TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(TEST_SET_PATH, index=False)
    logger.info(f"Untouched real test set safely persisted to: {TEST_SET_PATH}")

    return train_df, test_df, label_mapping


if __name__ == "__main__":
    train_df, test_df, mapping = load_and_preprocess_raw_data()
    print("Preprocessing completed successfully.")
