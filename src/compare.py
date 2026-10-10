from pathlib import Path

import pandas as pd

# Works from any location: src/compare.py -> assignment-1/
BASE_DIR = Path(__file__).resolve().parent.parent / "assignment-1"

FIELDS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]
ID_COLUMN = "File Name"


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Tolerate "file_name" vs "File Name".
    df = df.rename(columns={c: ID_COLUMN for c in df.columns
                            if c.strip().lower().replace("_", " ") == "file name"})
    return df


truth = load(BASE_DIR / "data" / "test.csv")
pred = load(BASE_DIR / "outputs" / "test_predictions.csv")

# Predictions may cover only a subset of documents (e.g. a partial run).
merged = truth.merge(pred, on=ID_COLUMN, suffixes=("_true", "_pred"), how="inner",
                     validate="one_to_one")
n = len(merged)
print(f"Documents compared: {n} of {len(truth)} in test.csv\n")


def trim(value):
    return str(value).strip()


def to_number(value):
    try:
        return float(trim(value).replace(",", ""))
    except ValueError:
        return None


print("RAW EXACT-MATCH RECALL (no normalization)")
print("-" * 52)
for field in FIELDS:
    ok = (merged[f"{field}_true"] == merged[f"{field}_pred"]).sum()
    print(f"{field:<24} {ok / n:>7.2%}  ({ok}/{n})")

print("\nWHITESPACE-TRIMMED EXACT-MATCH RECALL (leading/trailing spaces ignored)")
print("-" * 52)
for field in FIELDS:
    ok = (merged[f"{field}_true"].map(trim) == merged[f"{field}_pred"].map(trim)).sum()
    print(f"{field:<24} {ok / n:>7.2%}  ({ok}/{n})")

print("\nNUMERIC-EQUIVALENCE DIAGNOSTIC (analysis only, not the assignment score)")
print("-" * 52)
for field in ["Aggrement Value", "Renewal Notice (Days)"]:
    ok = 0
    for a, p in zip(merged[f"{field}_true"], merged[f"{field}_pred"]):
        a_num, p_num = to_number(a), to_number(p)
        ok += a_num is not None and p_num is not None and a_num == p_num
    print(f"{field:<24} {ok / n:>7.2%}  ({ok}/{n})")

print("\nMISMATCHES (after trimming whitespace)")
print("-" * 52)
for _, row in merged.iterrows():
    for field in FIELDS:
        a, p = trim(row[f"{field}_true"]), trim(row[f"{field}_pred"])
        if a != p:
            print(f"{row[ID_COLUMN]} | {field} | expected={a!r} | predicted={p!r}")
