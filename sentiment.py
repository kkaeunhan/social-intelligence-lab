from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable, Any

from utils import (
    quadratic_weighted_kappa,
    read_json,
    write_json,
    write_timestamped_json,
)


def preprocess_content(content: str) -> str:
    """Preprocess one review content before sentiment prediction."""
    text = html.unescape(str(content)).strip()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", text)
    text = re.sub(r"\b(?:\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})\b", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    text = re.sub(r"([!?])\1{2,}", r"\1\1\1", text)
    text = re.sub(r"(\.)\1{2,}", "...", text)
    text = re.sub(r"(ㅋ)\1{4,}", "ㅋㅋㅋㅋ", text)
    text = re.sub(r"(ㅎ)\1{4,}", "ㅎㅎㅎㅎ", text)
    text = re.sub(r"([ㅠㅜ])\1{4,}", r"\1\1\1\1", text)
    text = re.sub(r"([가-힣])\1{3,}", r"\1\1", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def predict_sentiment(content: str) -> int:
    """Predict a sentiment label for one review.

    Labels: 0=negative, 1=weak_negative, 2=weak_positive, 3=positive
    """
    # TODO: Implement your sentiment classification logic.
    raise NotImplementedError("predict_sentiment is not implemented yet.")


def make_prediction(
    review_id: str | int,
    content: str,
    predict_func: Callable[[str], int] = predict_sentiment,
) -> dict[str, str | int]:
    """Create one prediction item with the required output schema."""
    return {
        "review_id": review_id,
        "pred": predict_func(content),
    }


def load_review_data(file_path: str | Path) -> list[dict[str, Any]]:
    """Load review data."""
    records = read_json(file_path)

    if any("review_id" not in record for record in records):
        raise ValueError("Every record must have a review_id.")

    return records


def make_labels(records: list[dict[str, Any]]) -> list[dict[str, str | int]]:
    return [
        {"review_id": record["review_id"], "label": record["label"]}
        for record in records
        if "label" in record
    ]


def evaluate(
    predictions: list[dict[str, str | int]],
    labels: list[dict[str, str | int]],
) -> float:
    """Evaluate predictions with quadratic weighted kappa."""
    pred_by_id = {item["review_id"]: int(item["pred"]) for item in predictions}
    label_by_id = {item["review_id"]: int(item["label"]) for item in labels}

    if set(pred_by_id) != set(label_by_id):
        raise ValueError("prediction and label review_id sets must match.")

    review_ids = sorted(label_by_id)
    y_true = [label_by_id[review_id] for review_id in review_ids]
    y_pred = [pred_by_id[review_id] for review_id in review_ids]

    return float(quadratic_weighted_kappa(y_true, y_pred, num_classes=4))


def run_sentiment_analysis(
    file_path: str | Path,
    output_dir: str | Path = "output",
    preprocess_func: Callable[[str], str] = preprocess_content,
    predict_func: Callable[[str], int] = predict_sentiment,
) -> dict[str, Any]:
    """Run sentiment prediction for a JSON review file and save the result."""
    records = load_review_data(file_path)
    predictions = []

    for record in records:
        content = preprocess_func(record["content"])
        predictions.append(make_prediction(record["review_id"], content, predict_func))

    labels = make_labels(records)
    score = evaluate(predictions, labels) if labels else None
    result = {
        "input_file": str(file_path),
        "num_records": len(records),
        "score": score,
        "evaluation": {
            "metric": "quadratic_weighted_kappa",
            "score": score,
        },
        "predictions": predictions,
    }
    output_path = write_timestamped_json(
        result,
        output_dir=output_dir,
        prefix="sentiment_predictions",
    )
    result["output_path"] = str(output_path)
    write_json(result, output_path)
    return result
