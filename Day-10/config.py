from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

# Dataset configuration
DATASET_PATH = RAW_DATA_DIR / "titanic.csv"
TARGET_COLUMN = "survived"

# Machine learning configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
CV_FOLDS = 5

# Model configuration
MODEL_PATH = MODEL_DIR / "best_model.joblib"

# Output files
METRICS_PATH = REPORT_DIR / "metrics.json"
MODEL_COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
ERROR_ANALYSIS_PATH = REPORT_DIR / "error_analysis.csv"

# Visualization settings
FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)
