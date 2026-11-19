"""Shared settings for every script in this project."""
from pathlib import Path

STUDENT_USN = "1MS25CI148"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# Local SQLite store: no separate server needed, and the model registry works.
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "agnews-topic-champion-challenger"
REGISTERED_MODEL = "news-topic-model"

# AG News label order (same order the dataset uses: 0..3)
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

SAMPLES_PER_CLASS = 100   # 4 x 100 = 400 evaluation articles
SEED = 148
MAX_LENGTH = 256
BATCH_SIZE = 16

def to_class(label: str) -> str:
    """Map a model's raw label (e.g. 'LABEL_2') to the AG News class name."""
    if label in CLASSES:
        return label
    return CLASSES[int(str(label).rsplit("_", 1)[-1])]


MODELS = {
    # Current production model: small and fast (~12M params)
    "champion": {"short": "albert-agnews", "hf_id": "textattack/albert-base-v2-ag-news"},
    # Candidate: much bigger (~125M params)
    "challenger": {"short": "roberta-agnews", "hf_id": "textattack/roberta-base-ag-news"},
}
