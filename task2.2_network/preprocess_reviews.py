from __future__ import annotations

import argparse
import html
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "si_dataset" / "review_for_analysis.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "task2.2_network" / "data"
DEFAULT_MIN_CHARS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Coupang reviews and build analysis_text for network analysis."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() == "nan"


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""

    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", text)
    text = re.sub(r"\b(?:\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})\b", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"(ㅋ)\1{4,}", "ㅋㅋㅋㅋ", text)
    text = re.sub(r"(ㅎ)\1{4,}", "ㅎㅎㅎㅎ", text)
    text = re.sub(r"([!?])\1{2,}", r"\1\1\1", text)
    text = re.sub(r"(\.)\1{2,}", "...", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_identifier(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def safe_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sentiment_from_rating(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    if rating > 0:
        return "negative"
    return "unknown"


def meaningful_char_count(text: str) -> int:
    return len(re.findall(r"[0-9A-Za-z가-힣]", text))


def clean_survey_answers(record: dict[str, Any]) -> list[dict[str, str]]:
    answers: list[dict[str, str]] = []
    for survey in record.get("reviewSurveyAnswers") or []:
        if not isinstance(survey, dict):
            continue
        question = clean_text(survey.get("question"))
        answer = clean_text(survey.get("answer"))
        if question or answer:
            answers.append({"question": question, "answer": answer})
    return answers


def build_analysis_text(title: str, content: str, survey_text: str) -> str:
    fields = [
        ("제목", title),
        ("본문", content),
        ("설문", survey_text),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def normalize_record(record: dict[str, Any], min_chars: int) -> tuple[dict[str, Any] | None, str]:
    review_id = clean_identifier(record.get("reviewId") or record.get("review_id"))
    product_name = clean_text(record.get("product_name"))
    title = clean_text(record.get("title"))
    content = clean_text(record.get("content"))
    item_name = clean_text(record.get("itemName") or record.get("item_name"))
    survey_answers = clean_survey_answers(record)
    survey_text = " / ".join(
        f"{answer['question']}: {answer['answer']}".strip(": ")
        for answer in survey_answers
        if answer.get("question") or answer.get("answer")
    )
    analysis_text = build_analysis_text(title, content, survey_text)
    analysis_char_len = len(analysis_text)
    analysis_meaningful_chars = meaningful_char_count(analysis_text)

    if not review_id:
        return None, "missing_review_id"
    if not product_name:
        return None, "missing_product_name"
    if not analysis_text:
        return None, "missing_analysis_text"
    if analysis_char_len < min_chars or analysis_meaningful_chars < min_chars:
        return None, "short_analysis_text"

    rating = safe_int(record.get("rating"), 0)
    cleaned = {
        "review_id": review_id,
        "source_review_id": review_id,
        "product_name": product_name,
        "product_id": safe_int(record.get("productId"), 0),
        "vendor_item_id": safe_int(record.get("vendorItemId"), 0),
        "item_id": safe_int(record.get("itemId"), 0),
        "item_name": item_name,
        "title": title,
        "content": content,
        "survey_answers": survey_answers,
        "survey_text": survey_text,
        "analysis_text": analysis_text,
        "analysis_char_len": analysis_char_len,
        "analysis_meaningful_chars": analysis_meaningful_chars,
        "rating": rating,
        "sentiment_id": sentiment_from_rating(rating),
        "review_at": safe_int(record.get("reviewAt"), 0),
        "created_at": safe_int(record.get("createdAt"), 0),
        "helpful_count": safe_int(record.get("helpfulCount"), 0),
        "helpful_true_count": safe_int(record.get("helpfulTrueCount"), 0),
        "helpful_false_count": safe_int(record.get("helpfulFalseCount"), 0),
        "has_image": len(record.get("attachments") or []) > 0,
        "has_video": len(record.get("videoAttachments") or []) > 0,
    }
    return cleaned, "kept"


def content_fingerprint(document: dict[str, Any]) -> str:
    content = document["analysis_text"].lower()
    content = "".join(content.split())
    return "".join(ch for ch in content if ch.isalnum() or "가" <= ch <= "힣")


def deduplicate_documents(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    seen_review_ids: set[str] = set()
    seen_content: set[str] = set()
    kept: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for document in documents:
        review_id = document["review_id"]
        fingerprint = content_fingerprint(document)
        if review_id in seen_review_ids:
            skipped["duplicate_review_id"] += 1
            continue
        if fingerprint in seen_content:
            skipped["duplicate_analysis_text"] += 1
            continue
        seen_review_ids.add(review_id)
        seen_content.add(fingerprint)
        kept.append(document)

    return kept, skipped


def write_jsonl(path: Path, documents: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8") as file:
        records = json.load(file)

    raw_skip_reasons: Counter[str] = Counter()
    normalized_documents: list[dict[str, Any]] = []
    for record in records:
        document, reason = normalize_record(record, args.min_chars)
        raw_skip_reasons[reason] += 1
        if document is not None:
            normalized_documents.append(document)

    cleaned_documents, duplicate_skip_reasons = deduplicate_documents(normalized_documents)
    for index, document in enumerate(cleaned_documents, start=1):
        document["row_id"] = index

    output_jsonl = args.output_dir / "cleaned_reviews.jsonl"
    output_json = args.output_dir / "cleaned_reviews_sample.json"
    output_summary = args.output_dir / "preprocess_summary.json"

    write_jsonl(output_jsonl, cleaned_documents)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(cleaned_documents[:20], file, ensure_ascii=False, indent=2)

    product_counts = Counter(document["product_name"] for document in cleaned_documents)
    sentiment_counts = Counter(document["sentiment_id"] for document in cleaned_documents)
    rating_counts = Counter(str(document["rating"]) for document in cleaned_documents)
    summary = {
        "input_path": str(args.input),
        "output_jsonl": str(output_jsonl),
        "sample_path": str(output_json),
        "raw_records": len(records),
        "normalized_before_dedup": len(normalized_documents),
        "cleaned_records": len(cleaned_documents),
        "min_chars": args.min_chars,
        "raw_skip_reasons": dict(raw_skip_reasons),
        "dedup_skip_reasons": dict(duplicate_skip_reasons),
        "product_counts": dict(sorted(product_counts.items())),
        "sentiment_counts": dict(sorted(sentiment_counts.items())),
        "rating_counts": dict(sorted(rating_counts.items(), key=lambda item: int(item[0]))),
        "analysis_char_len": {
            "min": min((d["analysis_char_len"] for d in cleaned_documents), default=0),
            "max": max((d["analysis_char_len"] for d in cleaned_documents), default=0),
            "avg": round(
                sum(d["analysis_char_len"] for d in cleaned_documents) / len(cleaned_documents),
                2,
            )
            if cleaned_documents
            else 0,
        },
    }
    with output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
