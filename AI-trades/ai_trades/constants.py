"""Stage-1 constants. Later stages must not retune these from test-period returns."""

from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

FEATURE_SOURCE = PROJECT_ROOT / "data" / "asset_indicator_matrix.csv"
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "models"
REPORT_DIR = PACKAGE_ROOT / "reports"

MODEL_VERSION = "stage1_train2018_v1"
MANIFEST_NAME = "manifest.jsonl"

# Inclusive cutoff. Rows after this date are held out of the stage-1 fit.
TRAIN_END = pd.Timestamp("2018-12-31")

# Marchenko-Pastur gamma in this module is n_features / n_observations,
# matching the usual (1 + sqrt(p/n))^2 edge when p < n.
CUMULATIVE_VARIANCE_TARGETS = (0.80, 0.90, 0.95)
ZERO_VARIANCE_TOLERANCE = 1e-12
EIGENVALUE_ATOL = 1e-6
