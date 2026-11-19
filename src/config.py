"""Central configuration file for Encrypted Network Traffic Classification System.

Stores paths, random seeds, selected model hyperparameters, and constants.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# File Paths
TEST_SET_PATH = PROCESSED_DATA_DIR / "test_set.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "selected_features.pkl"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Target Raw CSV filenames (CICIDS2017 4 files)
RAW_CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]

# Pipeline Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K_FEATURES = 20
TUNING_SAMPLE_SIZE = 50000
CV_FOLDS = 3
TARGET_COLUMN = "Label"
