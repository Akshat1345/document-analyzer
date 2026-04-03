"""Self-evaluation harness that mirrors hackathon scoring rubric."""

from __future__ import annotations

import base64
import json
import os
import re
from collections import Counter
from pathlib import Path

import httpx
from fuzzywuzzy import fuzz

API_URL = "http://localhost:8000/api/document-analyze"
API_KEY = os.getenv("API_KEY", "sk_track2_your_secret_key_here")


def score_summary(generated: str, reference_text: str) -> float:
    """Score summary in [0,2] based on overlap with top reference keywords."""

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", reference_text.lower())
    filtered = [token for token in tokens if len(token) > 3]
    counts = Counter(filtered)
    key_words = [word for word, _ in counts.most_common(10)]
    if not key_words:
        return 0.0

    generated_lower = generated.lower()
    overlap = [word for word in key_words if word in generated_lower]
    return min(2.0, (len(overlap) / len(key_words)) * 2)


def _fuzzy_f1(predicted: list[str], expected: list[str]) -> float:
    """Compute fuzzy-match F1 for one entity list."""

    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    matched_expected = set()
    true_positives = 0

    for pred in predicted:
        for idx, exp in enumerate(expected):
            if idx in matched_expected:
                continue
            if fuzz.ratio(pred.lower(), exp.lower()) > 80:
                matched_expected.add(idx)
                true_positives += 1
                break

    precision = true_positives / max(1, len(predicted))
    recall = true_positives / max(1, len(expected))
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def score_entities(extracted_entities: dict, ground_truth_entities: dict) -> float:
    """Score entities in [0,4] by averaging per-field fuzzy F1."""

    fields = ["names", "dates", "organizations", "amounts"]
    f1_values = []
    for field in fields:
        predicted = extracted_entities.get(field, [])
        expected = ground_truth_entities.get(field, [])
        f1_values.append(_fuzzy_f1(predicted, expected))

    avg_f1 = sum(f1_values) / len(fields)
    return avg_f1 * 4.0


def score_sentiment(predicted: str, actual: str) -> float:
    """Score sentiment in [0,4] using exact match."""

    return 4.0 if predicted == actual else 0.0


def main() -> None:
    """Run full scoring workflow and print report."""

    eval_dir = Path(__file__).resolve().parent
    ground_truth_path = eval_dir / "ground_truth.json"

    with ground_truth_path.open("r", encoding="utf-8") as f:
        test_cases = json.load(f)

    max_raw = len(test_cases) * 10
    total_raw = 0.0

    print("=" * 72)
    print("DocuMind AI Evaluation Report")
    print("=" * 72)

    for case in test_cases:
        file_name = case["fileName"]
        file_type = case["fileType"]
        file_path = eval_dir.parent / "test_files" / file_name

        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        with file_path.open("rb") as file_handle:
            encoded = base64.b64encode(file_handle.read()).decode("utf-8")

        payload = {"fileName": file_name, "fileType": file_type, "fileBase64": encoded}

        try:
            response = httpx.post(
                API_URL,
                headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
        except Exception as exc:
            raise RuntimeError(
                "API unreachable at http://localhost:8000. Start it first using docker-compose up --build."
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(f"API request failed for {file_name}: {response.status_code} {response.text}")

        result = response.json()
        if result.get("status") != "success":
            raise RuntimeError(f"API returned error for {file_name}: {result}")

        reference_text = " ".join(
            case["expected_entities"].get("organizations", []) + case["expected_entities"].get("names", [])
        )

        summary_score = score_summary(result.get("summary", ""), reference_text)
        entities_score = score_entities(result.get("entities", {}), case["expected_entities"])
        sentiment_score = score_sentiment(result.get("sentiment", "Neutral"), case["expected_sentiment"])
        case_total = summary_score + entities_score + sentiment_score
        total_raw += case_total

        print(f"File: {file_name}")
        print(f"  Summary score:   {summary_score:.2f} / 2.00")
        print(f"  Entities score:  {entities_score:.2f} / 4.00")
        print(f"  Sentiment score: {sentiment_score:.2f} / 4.00")
        print(f"  Case total:      {case_total:.2f} / 10.00")
        print("-" * 72)

    projected_90 = (total_raw / max_raw) * 90 if max_raw else 0.0
    print(f"Total raw score: {total_raw:.2f} / {max_raw}")
    print(f"Projected final score: {projected_90:.2f} / 90")


if __name__ == "__main__":
    main()
