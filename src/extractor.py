
import json
import os
import time

import requests

BACKEND = os.environ.get("BACKEND", "ollama").strip().lower()

# --- Ollama (local) settings ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_NUM_THREAD = int(os.environ.get("OLLAMA_NUM_THREAD", "0"))  # 0 = let Ollama decide
NUM_CTX = 4096
NUM_PREDICT = 200
KEEP_ALIVE = "30m"

# --- Google Gemini (hosted) settings ---
# gemini-3.5-flash-lite: fastest/cheapest stable model. For higher accuracy try

GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash-lite")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

MODEL_NAME = GOOGLE_MODEL if BACKEND == "google" else OLLAMA_MODEL

FIELDS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]

MAX_EXAMPLE_CHARS = 2500
MAX_DOC_CHARS = 8000

SCHEMA = {
    "type": "object",
    "properties": {field: {"type": "string"} for field in FIELDS},
    "required": FIELDS,
}

_session = requests.Session()


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------
def _format_examples(examples):
    blocks = []
    for i, example in enumerate(examples or [], start=1):
        text = example["text"][:MAX_EXAMPLE_CHARS]
        blocks.append(
            f"Example {i} document:\n{text}\n"
            f"Example {i} labels:\n{json.dumps(example['labels'], ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


def _build_prompt(document_text, examples):
    example_block = _format_examples(examples)
    document_text = document_text[:MAX_DOC_CHARS]
    return f"""Extract metadata from the rental agreement below.
Return one JSON object with exactly these keys: {json.dumps(FIELDS)}
All values must be strings.

Rules:
- Aggrement Value: the monthly rent as a whole number in digits only (no commas, currency symbols, words or decimal part; for example "Rs. 15,000.00" becomes 15000). Not the deposit or advance.
- Aggrement Start Date: the date the tenancy begins.
- Aggrement End Date: the last day of the rental term. Use the stated expiry date if there is one. If the document only gives a duration, work the final day out from the start date (for example, 12 months from 01.06.2005 ends on 31.05.2006). Always give your best answer; do not leave this empty.
- Renewal Notice (Days): the notice period for terminating or vacating, in days as digits (use 30 days per month, so "two months" is 60). Leave empty only if the agreement has no notice clause at all.
- Party One: the first-named party (usually the owner or lessor). Party Two: the second-named party (usually the tenant or lessee).
- Party names: the name only. No titles (Sri, Shri, Mr, Mrs, Ms), no S/o or D/o part, no address, no role. Copy the spelling, initials, punctuation and spacing exactly as written; do not abbreviate or expand anything. Leave out stray punctuation (such as a trailing comma) that only separates the name from the text after it.
- Write dates in the same format as the example labels.
- Never invent values.
- The example only shows the labeling style; never copy its values.

{example_block}

DOCUMENT:
{document_text}

Return only the JSON object."""


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------
def _ollama_post(prompt, output_format):
    options = {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}
    if OLLAMA_NUM_THREAD > 0:
        options["num_thread"] = OLLAMA_NUM_THREAD
    response = _session.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": output_format,
            "keep_alive": KEEP_ALIVE,
            "options": options,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _call_ollama(prompt):
    try:
        return _ollama_post(prompt, SCHEMA)
    except requests.HTTPError:
        # Older Ollama versions do not accept a JSON schema; use plain JSON mode.
        return _ollama_post(prompt, "json")


def _call_google(prompt):
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set. Run: export GOOGLE_API_KEY=your-key")

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            # generous: some Gemini models count internal "thinking" tokens in this limit
            "maxOutputTokens": 2048,
        },
    }
    headers = {"x-goog-api-key": GOOGLE_API_KEY, "Content-Type": "application/json"}
    url = GOOGLE_URL.format(model=GOOGLE_MODEL)

    last_error = None
    for attempt in range(5):
        response = _session.post(url, headers=headers, json=body, timeout=120)
        if response.status_code in (429, 500, 503):  # rate limit / temporary overload
            last_error = f"{response.status_code}: {response.text[:200]}"
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError(f"Gemini returned no candidates: {str(data)[:300]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)
    raise RuntimeError(f"Gemini request failed after retries ({last_error})")


def _parse_json(raw):
    """Parse a JSON object, tolerating markdown fences or text around it."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"Model returned no JSON object: {raw[:300]}")
    try:
        result = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {raw[:300]}") from exc
    if isinstance(result, dict) and isinstance(result.get("json"), dict):
        result = result["json"]
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object, got {type(result).__name__}")
    return result


def _clean_value(value):
    if value is None:
        return None
    cleaned = str(value).strip()
    if cleaned.lower() in {"", "null", "none", "unknown", "n/a"}:
        return None
    return cleaned


def extract_metadata(document_text, examples=None):
    """Extract all six fields with ONE model call."""
    prompt = _build_prompt(document_text, examples)
    raw = _call_google(prompt) if BACKEND == "google" else _call_ollama(prompt)
    result = _parse_json(raw)
    return {field: _clean_value(result.get(field)) for field in FIELDS}


if __name__ == "__main__":
    print(f"Backend: {BACKEND} | Model: {MODEL_NAME}")
    if BACKEND == "google":
        print("GOOGLE_API_KEY set:", bool(GOOGLE_API_KEY))
    else:
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        r.raise_for_status()
        names = [m.get("name") for m in r.json().get("models", [])]
        print("Ollama is reachable. Models:", names)
        if not any(n and n.startswith(OLLAMA_MODEL) for n in names):
            print(f"WARNING: {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}")