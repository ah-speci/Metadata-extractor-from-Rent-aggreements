from pathlib import Path
import json
import sys
import time
import traceback

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Expected layout:
# assignment1/
#   src/predict.py
#   src/extractor.py
#   src/compare.py
#   assignment-1/data/{train.csv,test.csv}
#   assignment-1/outputs/extracted_text.csv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR / "assignment-1"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TEXT_PATH = OUTPUT_DIR / "extracted_text.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "prediction_diagnostics.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extractor import BACKEND, MODEL_NAME, extract_metadata  # noqa: E402

FIELDS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]
ID_COLUMN = "File Name"  # same column name as test.csv, so compare.py can merge on it
NUM_EXAMPLES = 1         # one example keeps the prompt short and inference fast


def normalize_value(value):
    """Convert missing values to empty strings and trim surrounding whitespace."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    if value.lower() in {"", "none", "null", "nan", "n/a", "unknown"}:
        return ""
    return value


def normalize_column_name(column):
    return str(column).strip().lower().replace("_", " ").replace("-", " ")


def find_column(df, candidates, what):
    for column in df.columns:
        if normalize_column_name(column) in candidates:
            return column
    raise ValueError(f"Could not identify the {what} column. Available: {list(df.columns)}")


def standardize_id_column(df):
    column = find_column(
        df, {"file name", "filename", "file", "document", "document id", "id"}, "document ID"
    )
    df = df.rename(columns={column: ID_COLUMN})
    df[ID_COLUMN] = df[ID_COLUMN].astype(str).str.strip()
    return df


def normalize_filename(filename):
    """Strip known extensions so document IDs match extracted-text IDs."""
    name = Path(str(filename)).name.strip().lower()
    for suffix in (".pdf.docx", ".docx", ".png", ".jpg", ".jpeg", ".pdf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def build_example_pool(train_df, train_text_lookup, test_ids):
    """Labeled training examples, excluding documents that also appear in test."""
    pool, seen = [], set()
    for _, row in train_df.iterrows():
        key = normalize_filename(row[ID_COLUMN])
        if key in test_ids:
            continue
        text = train_text_lookup.get(key, "")
        if not text:
            continue
        signature = " ".join(text.split()).casefold()
        if signature in seen:
            continue
        seen.add(signature)
        pool.append({
            "file_name": row[ID_COLUMN],
            "text": text,
            "labels": {f: normalize_value(row.get(f, "")) for f in FIELDS if f in train_df.columns},
        })
    return pool


class ExampleRetriever:
    """TF-IDF index built once over the example pool (not once per document)."""

    def __init__(self, pool):
        self.pool = pool
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=20000, sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([e["text"] for e in pool]) if pool else None

    def top_k(self, text, k):
        if not self.pool:
            return []
        scores = cosine_similarity(self.vectorizer.transform([text]), self.matrix).flatten()
        order = scores.argsort()[::-1][:k]
        return [self.pool[i] for i in order if scores[i] > 0]


def save_results(predictions, diagnostics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions, columns=[ID_COLUMN] + FIELDS).to_csv(PREDICTIONS_PATH, index=False)
    pd.DataFrame(diagnostics).to_csv(DIAGNOSTICS_PATH, index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_PATH, TEST_PATH, TEXT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_df = standardize_id_column(pd.read_csv(TRAIN_PATH, dtype=str).fillna(""))
    test_df = standardize_id_column(pd.read_csv(TEST_PATH, dtype=str).fillna(""))
    text_df = standardize_id_column(pd.read_csv(TEXT_PATH, dtype=str).fillna(""))
    text_column = find_column(
        text_df, {"text", "extracted text", "document text", "content"}, "document text"
    )

    if "split" in text_df.columns:
        split = text_df["split"].astype(str).str.lower()
        train_text_df, test_text_df = text_df[split == "train"], text_df[split == "test"]
    else:
        train_text_df = test_text_df = text_df

    def make_lookup(frame):
        lookup = {}
        for _, row in frame.iterrows():
            lookup.setdefault(normalize_filename(row[ID_COLUMN]), normalize_value(row[text_column]))
        return lookup

    train_lookup, test_lookup = make_lookup(train_text_df), make_lookup(test_text_df)
    test_ids = {normalize_filename(n) for n in test_df[ID_COLUMN]}

    pool = build_example_pool(train_df, train_lookup, test_ids)
    retriever = ExampleRetriever(pool)
    print(f"Backend: {BACKEND} | Model: {MODEL_NAME}")
    print(f"Loaded {len(pool)} labeled examples.")

    # Optional: python src/predict.py 2   -> only process the first 2 test documents
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(test_df)

    predictions, diagnostics = [], []
    run_start = time.time()

    for index, row in test_df.head(limit).iterrows():
        document_id = row[ID_COLUMN]
        text = test_lookup.get(normalize_filename(document_id), "")
        print(f"\n[{index + 1}/{limit}] {document_id} ({len(text)} chars)", flush=True)

        started = time.time()
        if not text:
            prediction, status = {f: "" for f in FIELDS}, "missing_text"
        else:
            examples = [
                {"text": e["text"], "labels": e["labels"]}
                for e in retriever.top_k(text, NUM_EXAMPLES)
            ]
            try:
                extracted = extract_metadata(document_text=text, examples=examples)
                prediction = {f: normalize_value(extracted.get(f)) for f in FIELDS}
                status = "ok" if any(prediction.values()) else "model_returned_all_empty"
            except Exception as exc:
                print(f"  Extraction failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                prediction, status = {f: "" for f in FIELDS}, "metadata_extraction_error"

        seconds = round(time.time() - started, 1)
        predictions.append({ID_COLUMN: document_id, **prediction})
        diagnostics.append({
            "file_name": document_id,
            "status": status,
            "seconds": seconds,
            "notice_prediction": prediction[FIELDS[3]],
        })
        print(f"  Done in {seconds}s ({status})")
        print(json.dumps(prediction, indent=2, ensure_ascii=False))
        save_results(predictions, diagnostics)  # saved after every document

    print(f"\nPrediction run complete in {round(time.time() - run_start, 1)}s.")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Diagnostics: {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
