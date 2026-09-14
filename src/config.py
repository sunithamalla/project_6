from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
STORE_PATH = DATA_DIR / "store.csv"

MODEL_PATH = ARTIFACT_DIR / "rossmann_lgbm.joblib"
META_PATH = ARTIFACT_DIR / "model_meta.json"
VALIDATION_PRED_PATH = ARTIFACT_DIR / "validation_predictions.csv"
METRICS_PATH = ARTIFACT_DIR / "validation_metrics.json"
SUBMISSION_PATH = ARTIFACT_DIR / "submission.csv"

ARTIFACT_DIR.mkdir(exist_ok=True)
