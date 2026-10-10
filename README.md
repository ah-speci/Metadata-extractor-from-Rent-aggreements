# Metadata Extraction from Rental Agreements

An LLM-based system that extracts six metadata fields from rental agreements, regardless of template.
It uses no regular expressions or rule-based logic: a language model reads the document text, guided by
one similar labeled example retrieved from the training set.

| Field | Meaning |
|---|---|
| `Aggrement Value` | Monthly rent (whole number) |
| `Aggrement Start Date` | Date the rental term begins |
| `Aggrement End Date` | Last day of the rental term |
| `Renewal Notice (Days)` | Termination / vacating notice period in days |
| `Party One` | First contracting party (name only) |
| `Party Two` | Second contracting party (name only) |

The field names keep the assignment's original spelling ("Aggrement").

## Approach

1. **Text extraction.** Each test and training document (DOCX or scanned image) is converted to plain text and stored
   in `assignment-1/outputs/extracted_text.csv`.
   *[TODO: describe the script/tool used here (e.g. DOCX parsing, OCR for PNG) and include it in the repo.]*
2. **Example retrieval.** A TF-IDF index (unigrams and bigrams, scikit-learn) is built once over the labeled training
   documents. For each test document, the most similar training document and its labels are retrieved
   and shown to the model as a single example of the dataset's labeling style. Training documents that also appear in the
   test set are excluded from the example pool (one overlap, `24158401-Rental-Agreement`, is excluded), so no test label is ever shown to
   the model for its own document.
3. **One LLM call per document.** A single prompt with the field definitions, the retrieved example and the document text
   asks for a JSON object with the six fields. Temperature is 0. The prompt gives *field definitions and labeling
   conventions* (monthly rent as a whole number, name only without titles, notice in days); it contains no
   per-document answers.
4. **Output.** Values are trimmed of surrounding whitespace and saved to `assignment-1/outputs/test_predictions.csv`
   after every document, together with timing and status in `prediction_diagnostics.csv`.

No post-processing rules (regex, string fixes, or static conditions) are applied to the model's answers.

### Backends

`src/extractor.py` supports two interchangeable backends, selected with the `BACKEND` environment variable.
Both use the same prompt.

| `BACKEND` | Model (default) | Needs |
|---|---|---|
| `ollama` (default) | `qwen2.5:3b` (`OLLAMA_MODEL` to change) | Ollama running locally, no key |
| `google` | `gemini-3.5-flash-lite` (`GOOGLE_MODEL` to change) | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) |

**The submitted predictions were produced with `BACKEND=google` and `GOOGLE_MODEL=gemini-3.8-flash`.**
API keys are read only from environment variables and are never stored in the code.

## Project layout

```
assignment1/
├── src/
│   ├── extractor.py   # prompt + backends (Ollama / Google Gemini)
│   ├── predict.py     # retrieval, extraction loop, saves predictions
│   └── compare.py     # per-field recall against test.csv
├── jupyter notebook/
│   └── eda.ipynb      # exploratory data analysis of the dataset
└── assignment-1/
    ├── data/          # train.csv, test.csv
    └── outputs/       # extracted_text.csv, test_predictions.csv, prediction_diagnostics.csv
```

## Setup

```
pip install pandas requests scikit-learn
```

**Local (no API key):**
```
ollama pull qwen2.5:3b
ollama serve                    # if not already running
python src/extractor.py         # checks that Ollama and the model are available
```

**Hosted Gemini:**
```
export GOOGLE_API_KEY=your-key
export BACKEND=google
export GOOGLE_MODEL=gemini-3.8-flash     # optional; default is gemini-3.5-flash-lite
```

## Reproducing the predictions

```
cd assignment1
python src/predict.py          # all test documents (add a number to limit, e.g. `python src/predict.py 2`)
python src/compare.py          # prints per-field recall and the mismatches
```

Predictions: `assignment-1/outputs/test_predictions.csv`.

## Results

Recall per field = exact matches / documents, on the 4 test documents.
**The raw exact-match score is the assignment's metric.** The whitespace-trimmed score is shown only as additional
analysis: several labels in `test.csv` contain leading or trailing spaces (for example `'Hanumaiah '`), which makes Party One
score 0% under raw matching even though the names are correct. We did not pad predictions with spaces to match.

**Final run: `gemini-3.8-flash`**

| Field | Raw exact match | Whitespace-trimmed |
|---|---|---|
| Aggrement Value | 100% (4/4) | 100% (4/4) |
| Aggrement Start Date | 100% (4/4) | 100% (4/4) |
| Aggrement End Date | 75% (3/4) | 75% (3/4) |
| Renewal Notice (Days) | 100% (4/4) | 100% (4/4) |
| Party One | 0% (0/4) | 100% (4/4) |
| Party Two | 0% (0/4) | 25% (1/4) |

**Comparison: `gemini-3.5-flash-lite`** (same code and prompt)

| Field | Raw exact match | Whitespace-trimmed |
|---|---|---|
| Aggrement Value | 100% | 100% |
| Aggrement Start Date | 75% | 75% |
| Aggrement End Date | 75% | 75% |
| Renewal Notice (Days) | 100% | 100% |
| Party One | 0% | 100% |
| Party Two | 0% | 25% |

Runtime for the 4 test documents: about 5 s (`gemini-3.5-flash-lite`), about 30 s (`gemini-3.8-flash`). The local
`qwen2.5:3b` run on a CPU-only machine took about 5 minutes and left several fields empty in an earlier prompt version; it was
not scored with the final prompt.

### Remaining errors (final run, after trimming whitespace)

| Document | Field | Expected | Predicted | Cause |
|---|---|---|---|---|
| 95980236 | End Date | `31.03.2011` | `28.02.2011` | The agreement says only "11 months commencing from 1 April 2010"; no end date is written. The label assumes a 12-month term, and training labels for the same wording are inconsistent (some are impossible dates such as `31.02.2011`). |
| 95980236 | Party Two | `V.V.Ravi Kian` | `V.V. Ravi Kian` | The document spells the name with a space; the label removes it. |
| 156155545 | Party Two | `VYSHNAVI DAIRY SPECIALITIES Private Ltd` | `SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd.` | Model kept the "SRI" prefix and the final period; training labels are inconsistent about titles. |
| 228094620 | Party Two | `.B.Kishore` | `B.Kishore` | The document reads "Mr.B.Kishore"; the label appears to be a leftover after removing "Mr". |

## Limitations

- **Small test set.** With 4 documents each field moves in 25-point steps, so the scores are coarse.
- **Label noise.** Several expected values cannot be derived consistently from the documents (see the table above), which caps
  achievable exact-match recall without fitting to the labels. We chose not to hard-code or post-process to match them.
- **Run-to-run variation.** Hosted models are not fully repeatable even at temperature 0. In one `gemini-3.5-flash-lite` run the
  start date of a document with the typo "7st July" was read as `01.07.2013`; another run and `gemini-3.8-flash` read it correctly.
- **Party-name conventions** (titles, punctuation, spacing) are only partly learnable from one retrieved example.
- **No REST API** is implemented (it was optional).
