from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_REVIEWS = ROOT / "data" / "raw" / "reviews.parquet"
CLEAN_REVIEWS = ROOT / "data" / "processed" / "reviews_clean.parquet"
LLM_LABELS = ROOT / "data" / "labels" / "llm_labels.jsonl"
GOLD_TEMPLATE = ROOT / "data" / "labels" / "gold_to_label.csv"
GOLD_LABELS = ROOT / "data" / "labels" / "gold_labels.csv"
LABELLED_REVIEWS = ROOT / "data" / "processed" / "reviews_labelled.parquet"
MODEL_DIR = ROOT / "models"
OUTPUTS = ROOT / "outputs"
CHARTS = OUTPUTS / "charts"
