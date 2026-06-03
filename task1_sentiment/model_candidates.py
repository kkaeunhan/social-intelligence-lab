"""Compare Task 1 sentiment model candidates.

Use this file as the experiment space for trying multiple models under the
same validation split and metrics.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentiment import load_review_data, preprocess_content
from utils import quadratic_weighted_kappa, read_json, write_json

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import Pipeline
except ImportError:
    TfidfVectorizer = None
    LogisticRegression = None
    accuracy_score = None
    f1_score = None
    Pipeline = None


DEFAULT_DATA_PATH = "si_dataset/train_review_data.json"
DEFAULT_OUTPUT_DIR = "output/task1_sentiment"
TRANSFORMER_MODEL_CANDIDATES = {
    "klue_roberta": "klue/roberta-base",
    "kr_electra": "snunlp/KR-ELECTRA-discriminator",
    "koelectra_base": "monologg/koelectra-base-v3-discriminator",
    "koelectra_small_nsmc": "daekeun-ml/koelectra-small-v3-nsmc",
    "koelectra_sentiment": "cringepnh/koelectra-korean-sentiment",
    "beomi_kc_electra": "beomi/KcELECTRA-base-v2022",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Task 1 sentiment data for model candidate experiments."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the labeled review data JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where prepared data and split metadata are saved.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Deprecated: validation ratio for the old holdout split.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for stratified k-fold evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible stratified k-fold split.",
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save preprocessed records for inspection and debugging.",
    )
    parser.add_argument(
        "--augmentation-paths",
        nargs="*",
        default=[],
        help=(
            "Optional generated review JSON files. These are appended only to "
            "each fold's training split; validation remains original-only."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tfidf_logreg"],
        choices=[
            "tfidf_logreg",
            "klue_roberta",
            "kr_electra",
            "koelectra_base",
            "koelectra_small_nsmc",
            "koelectra_sentiment",
            "beomi_kc_electra",
        ],
        help="Model candidates to train and evaluate.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare data and split metadata without training candidates.",
    )
    parser.add_argument(
        "--transformer-epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs for transformer candidates.",
    )
    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=16,
        help="Batch size for transformer fine-tuning and evaluation.",
    )
    parser.add_argument(
        "--transformer-learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for transformer fine-tuning.",
    )
    parser.add_argument(
        "--transformer-max-length",
        type=int,
        default=256,
        help="Maximum token length for transformer tokenization.",
    )
    parser.add_argument(
        "--transformer-warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for transformer learning-rate scheduling.",
    )
    parser.add_argument(
        "--transformer-weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for transformer fine-tuning.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["inverse", "none", "sqrt_inverse", "effective_number", "custom"],
        default="inverse",
        help="Class weighting strategy for transformer loss.",
    )
    parser.add_argument(
        "--effective-number-beta",
        type=float,
        default=0.999,
        help="Beta for effective-number class weights.",
    )
    parser.add_argument(
        "--custom-class-weights",
        nargs=4,
        type=float,
        default=None,
        metavar=("W0", "W1", "W2", "W3"),
        help="Four custom class weights used when --class-weight-mode custom.",
    )
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "focal"],
        default="cross_entropy",
        help="Transformer loss function.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.0,
        help="Gamma for focal loss when --loss focal.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional suffix for output filenames to avoid overwriting experiments.",
    )
    parser.add_argument(
        "--ensemble-only",
        action="store_true",
        help="Evaluate an ensemble from saved validation prediction files only.",
    )
    parser.add_argument(
        "--ensemble-models",
        nargs="+",
        default=["klue_roberta", "koelectra_sentiment", "beomi_kc_electra"],
        help="Saved model names to combine for ensemble evaluation.",
    )
    parser.add_argument(
        "--ensemble-method",
        choices=["hard", "soft", "weighted_soft"],
        default="soft",
        help="Ensemble method. soft methods require saved class probabilities.",
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional model weights in the same order as --ensemble-models.",
    )
    return parser.parse_args()


def label_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(record["label"]) for record in records)
    return {str(label): counts.get(label, 0) for label in range(4)}


def prepare_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        prepared.append(
            {
                "review_id": record["review_id"],
                "product_name": record.get("product_name"),
                "content": record["content"],
                "clean_content": preprocess_content(record["content"]),
                "label": int(record["label"]),
                "label_string": record.get("label_string"),
            }
        )
    return prepared


def load_augmented_records(paths: list[str]) -> list[dict[str, Any]]:
    augmented_records = []
    seen_contents = set()

    for path in paths:
        payload = read_json(path)
        items = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(f"Augmentation file must contain a list or items: {path}")

        for index, record in enumerate(items):
            content = str(record.get("content", "")).strip()
            if not content:
                continue
            if content in seen_contents:
                continue
            seen_contents.add(content)
            label = int(record["label"])
            augmented_records.append(
                {
                    "review_id": record.get(
                        "review_id",
                        f"aug_label{label}_{len(augmented_records) + 1:06d}",
                    ),
                    "product_name": record.get("product_name"),
                    "content": content,
                    "clean_content": preprocess_content(content),
                    "label": label,
                    "label_string": record.get("label_string"),
                    "source": record.get("source", "llm_augmented"),
                    "is_augmented": True,
                    "augmentation_file": str(path),
                    "augmentation_index": index,
                }
            )

    return augmented_records


def append_augmented_train_records(
    folds: list[dict[str, Any]],
    augmented_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not augmented_records:
        return folds

    augmented_ids = {record["review_id"] for record in augmented_records}
    if len(augmented_ids) != len(augmented_records):
        raise ValueError("Augmented review_ids must be unique after normalization.")

    updated_folds = []
    for fold in folds:
        updated_fold = dict(fold)
        updated_fold["train_records"] = fold["train_records"] + augmented_records
        updated_fold["validation_records"] = fold["validation_records"]
        updated_fold["num_augmented_train_records"] = len(augmented_records)
        updated_folds.append(updated_fold)
    return updated_folds


def stratified_train_validation_split(
    records: list[dict[str, Any]],
    validation_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")

    records_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_label[int(record["label"])].append(record)

    rng = random.Random(seed)
    train_records = []
    validation_records = []

    for label in sorted(records_by_label):
        label_records = records_by_label[label]
        rng.shuffle(label_records)
        validation_count = max(1, round(len(label_records) * validation_size))
        validation_records.extend(label_records[:validation_count])
        train_records.extend(label_records[validation_count:])

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    return train_records, validation_records


def build_split_metadata(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    validation_size: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "strategy": "stratified_train_validation_split",
        "validation_size": validation_size,
        "num_train": len(train_records),
        "num_validation": len(validation_records),
        "train_label_distribution": label_distribution(train_records),
        "validation_label_distribution": label_distribution(validation_records),
        "train_review_ids": [record["review_id"] for record in train_records],
        "validation_review_ids": [record["review_id"] for record in validation_records],
    }


def stratified_kfold_split(
    records: list[dict[str, Any]],
    n_splits: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n_splits < 2:
        raise ValueError("--cv-folds must be at least 2.")

    records_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_label[int(record["label"])].append(record)

    min_label_count = min(len(label_records) for label_records in records_by_label.values())
    if n_splits > min_label_count:
        raise ValueError(
            f"--cv-folds={n_splits} is too large for the smallest class "
            f"count ({min_label_count})."
        )

    rng = random.Random(seed)
    validation_buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_splits)]

    for label in sorted(records_by_label):
        label_records = records_by_label[label][:]
        rng.shuffle(label_records)
        for index, record in enumerate(label_records):
            validation_buckets[index % n_splits].append(record)

    all_ids = {record["review_id"]: record for record in records}
    folds = []
    for fold_index, validation_records in enumerate(validation_buckets, start=1):
        validation_ids = {record["review_id"] for record in validation_records}
        train_records = [
            record
            for review_id, record in all_ids.items()
            if review_id not in validation_ids
        ]
        rng.shuffle(train_records)
        rng.shuffle(validation_records)
        folds.append(
            {
                "fold": fold_index,
                "train_records": train_records,
                "validation_records": validation_records,
            }
        )

    return folds


def build_kfold_metadata(
    folds: list[dict[str, Any]],
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "strategy": "stratified_kfold",
        "cv_folds": n_splits,
        "folds": [
            {
                "fold": fold["fold"],
                "num_train": len(fold["train_records"]),
                "num_validation": len(fold["validation_records"]),
                "num_augmented_train_records": int(
                    fold.get("num_augmented_train_records", 0)
                ),
                "train_label_distribution": label_distribution(fold["train_records"]),
                "validation_label_distribution": label_distribution(
                    fold["validation_records"]
                ),
                "train_review_ids": [
                    record["review_id"] for record in fold["train_records"]
                ],
                "validation_review_ids": [
                    record["review_id"] for record in fold["validation_records"]
                ],
            }
            for fold in folds
        ],
    }


def save_prepared_outputs(
    prepared_records: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_metadata = build_kfold_metadata(
        folds=folds,
        n_splits=args.cv_folds,
        seed=args.seed,
    )
    split_stem = f"stratified_kfold_seed{args.seed}_k{args.cv_folds}"
    if args.experiment_name:
        split_stem = f"{split_stem}_{args.experiment_name}"
    split_path = output_dir / f"{split_stem}.json"
    write_json(split_metadata, split_path)

    if args.save_preprocessed:
        preprocessed_stem = f"preprocessed_train_seed{args.seed}"
        if args.experiment_name:
            preprocessed_stem = f"{preprocessed_stem}_{args.experiment_name}"
        preprocessed_path = output_dir / f"{preprocessed_stem}.json"
        write_json(prepared_records, preprocessed_path)

    print("Saved split metadata:", split_path)
    print("Total label distribution:", label_distribution(prepared_records))
    for fold in split_metadata["folds"]:
        print(
            f"Fold {fold['fold']} validation label distribution:",
            fold["validation_label_distribution"],
        )


def require_sklearn() -> None:
    if any(
        item is None
        for item in (
            TfidfVectorizer,
            LogisticRegression,
            accuracy_score,
            f1_score,
            Pipeline,
        )
    ):
        raise ImportError("Install scikit-learn to train sklearn model candidates.")
    patch_optional_pandas_namespace()


def require_transformers() -> tuple[Any, ...]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise ImportError(
            "Install torch and transformers to train transformer candidates."
        ) from exc

    return (
        torch,
        DataLoader,
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )


def patch_optional_pandas_namespace() -> None:
    """Avoid sklearn optional pandas checks failing on a broken pandas namespace."""
    try:
        import pandas as pd
    except ImportError:
        return

    class _PandasPlaceholder:
        pass

    for name in ("DataFrame", "Series", "Index"):
        if not hasattr(pd, name):
            setattr(pd, name, _PandasPlaceholder)


def get_texts_and_labels(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[int]]:
    texts = [record["clean_content"] for record in records]
    labels = [int(record["label"]) for record in records]
    return texts, labels


def build_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int = 4,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[int(true_label)][int(pred_label)] += 1
    return matrix


def build_per_class_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int = 4,
) -> dict[str, dict[str, float | int]]:
    matrix = build_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_class = {}
    for label in range(num_classes):
        tp = matrix[label][label]
        fp = sum(matrix[row][label] for row in range(num_classes) if row != label)
        fn = sum(matrix[label][col] for col in range(num_classes) if col != label)
        support = sum(matrix[label])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
    return per_class


def evaluate_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if accuracy_score is None or f1_score is None:
        raise ImportError("Install scikit-learn to compute classification metrics.")

    return {
        "qwk": float(quadratic_weighted_kappa(y_true, y_pred, num_classes=4)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": build_confusion_matrix(y_true, y_pred),
        "per_class": build_per_class_metrics(y_true, y_pred),
        "label_distribution": {
            str(label): Counter(y_true).get(label, 0) for label in range(4)
        },
        "pred_distribution": {
            str(label): Counter(y_pred).get(label, 0) for label in range(4)
        },
    }


def build_validation_predictions(
    validation_records: list[dict[str, Any]],
    y_pred: list[int],
    y_prob: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, (record, pred) in enumerate(
        zip(validation_records, y_pred, strict=True)
    ):
        row = {
            "review_id": record["review_id"],
            "label": int(record["label"]),
            "pred": int(pred),
        }
        if y_prob is not None:
            row["probabilities"] = [float(value) for value in y_prob[index]]
        rows.append(row)
    return rows


def format_float_for_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def build_experiment_tag(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return args.experiment_name
    if args.loss == "cross_entropy" and args.class_weight_mode == "inverse":
        return ""

    parts = [args.loss, args.class_weight_mode]
    if args.class_weight_mode == "effective_number":
        parts.append(f"beta{format_float_for_tag(args.effective_number_beta)}")
    if args.loss == "focal":
        parts.append(f"gamma{format_float_for_tag(args.focal_gamma)}")
    if args.class_weight_mode == "custom" and args.custom_class_weights:
        weights = "_".join(format_float_for_tag(weight) for weight in args.custom_class_weights)
        parts.append(f"w{weights}")
    return "_".join(parts)


def append_experiment_tag(stem: str, args: argparse.Namespace) -> str:
    tag = build_experiment_tag(args)
    return f"{stem}_{tag}" if tag else stem


def save_validation_predictions(
    model_name: str,
    metrics: dict[str, Any],
    validation_records: list[dict[str, Any]],
    y_pred: list[int],
    args: argparse.Namespace,
    y_prob: list[list[float]] | None = None,
) -> Path:
    output = {
        "model_name": model_name,
        "split": f"stratified_{1 - args.validation_size:.0%}_{args.validation_size:.0%}_seed_{args.seed}",
        "metrics": metrics,
        "predictions": build_validation_predictions(validation_records, y_pred, y_prob),
    }
    output_path = (
        Path(args.output_dir)
        / f"validation_predictions_{append_experiment_tag(model_name, args)}_seed{args.seed}.json"
    )
    write_json(output, output_path)
    return output_path


def save_kfold_validation_predictions(
    model_name: str,
    aggregate_metrics: dict[str, float],
    fold_metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    args: argparse.Namespace,
    oof_metrics: dict[str, Any] | None = None,
) -> Path:
    output = {
        "model_name": model_name,
        "split": f"stratified_kfold_k{args.cv_folds}_seed_{args.seed}",
        "metrics": aggregate_metrics,
        "oof_metrics": oof_metrics,
        "fold_metrics": fold_metrics,
        "predictions": predictions,
    }
    output_path = (
        Path(args.output_dir)
        / f"validation_predictions_{append_experiment_tag(model_name, args)}_seed{args.seed}_k{args.cv_folds}.json"
    )
    write_json(output, output_path)
    return output_path


def summarize_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    summary = {}
    metric_names = ("qwk", "accuracy", "macro_f1")
    for metric_name in metric_names:
        values = [float(row["metrics"][metric_name]) for row in fold_metrics]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary[f"{metric_name}_mean"] = mean_value
        summary[f"{metric_name}_std"] = math.sqrt(variance)
    return summary


def save_model_comparison(
    comparison_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    output_path = (
        Path(args.output_dir)
        / f"model_comparison_{append_experiment_tag('transformer', args)}_seed{args.seed}_k{args.cv_folds}.json"
    )
    sorted_rows = sorted(
        comparison_rows,
        key=lambda row: row["metrics"]["qwk_mean"],
        reverse=True,
    )
    write_json(sorted_rows, output_path)
    return output_path


def train_tfidf_logreg(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    require_sklearn()

    train_texts, train_labels = get_texts_and_labels(train_records)
    validation_texts, validation_labels = get_texts_and_labels(validation_records)

    y_pred = fit_predict_tfidf_logreg(
        train_texts=train_texts,
        train_labels=train_labels,
        validation_texts=validation_texts,
        args=args,
    )
    metrics = evaluate_predictions(validation_labels, y_pred)
    predictions_path = save_validation_predictions(
        model_name="tfidf_logreg",
        metrics=metrics,
        validation_records=validation_records,
        y_pred=y_pred,
        args=args,
    )

    return {
        "model_name": "tfidf_logreg",
        "metrics": metrics,
        "validation_predictions_path": str(predictions_path),
        "config": get_tfidf_logreg_config(),
    }


def get_tfidf_logreg_config() -> dict[str, Any]:
    return {
        "vectorizer": {
            "type": "TfidfVectorizer",
            "analyzer": "char_wb",
            "ngram_range": [2, 5],
            "min_df": 2,
            "max_features": 120_000,
            "sublinear_tf": True,
        },
        "classifier": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 3_000,
            "solver": "lbfgs",
        },
    }


def fit_predict_tfidf_logreg(
    train_texts: list[str],
    train_labels: list[int],
    validation_texts: list[str],
    args: argparse.Namespace,
) -> list[int]:
    require_sklearn()

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=120_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3_000,
                    random_state=args.seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(train_texts, train_labels)
    return [int(pred) for pred in model.predict(validation_texts)]


def normalize_class_weights(weights: list[float]) -> list[float]:
    mean_weight = sum(weights) / len(weights)
    if mean_weight <= 0:
        raise ValueError("Class weights must have a positive mean.")
    return [float(weight) / mean_weight for weight in weights]


def make_class_weights(
    labels: list[int],
    mode: str = "inverse",
    effective_number_beta: float = 0.999,
    custom_class_weights: list[float] | None = None,
) -> list[float] | None:
    if mode == "none":
        return None

    if mode == "custom":
        if custom_class_weights is None or len(custom_class_weights) != 4:
            raise ValueError("--custom-class-weights requires exactly four values.")
        return normalize_class_weights([float(weight) for weight in custom_class_weights])

    counts = Counter(labels)
    num_labels = 4
    total = len(labels)

    if mode == "inverse":
        weights = [
            total / (num_labels * max(counts.get(label, 0), 1))
            for label in range(num_labels)
        ]
    elif mode == "sqrt_inverse":
        weights = [
            math.sqrt(total / (num_labels * max(counts.get(label, 0), 1)))
            for label in range(num_labels)
        ]
    elif mode == "effective_number":
        if not 0 < effective_number_beta < 1:
            raise ValueError("--effective-number-beta must be between 0 and 1.")
        weights = []
        for label in range(num_labels):
            count = max(counts.get(label, 0), 1)
            effective_num = (1 - effective_number_beta**count) / (
                1 - effective_number_beta
            )
            weights.append(total / (num_labels * effective_num))
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")

    return normalize_class_weights(weights)


def make_loss_fn(
    torch: Any,
    class_weights: Any,
    args: argparse.Namespace,
) -> Any:
    if args.loss == "cross_entropy":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    if args.loss == "focal":
        if args.focal_gamma < 0:
            raise ValueError("--focal-gamma must be non-negative.")

        def focal_loss(logits: Any, labels: Any) -> Any:
            ce_loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=class_weights,
                reduction="none",
            )
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** args.focal_gamma) * ce_loss
            return loss.mean()

        return focal_loss
    raise ValueError(f"Unknown loss: {args.loss}")


def get_loss_config(
    labels: list[int],
    class_weights: list[float] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "loss": args.loss,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "class_weight_mode": args.class_weight_mode,
        "effective_number_beta": (
            args.effective_number_beta
            if args.class_weight_mode == "effective_number"
            else None
        ),
        "custom_class_weights": args.custom_class_weights,
        "class_weights": class_weights,
        "train_label_distribution": {
            str(label): Counter(labels).get(label, 0) for label in range(4)
        },
        "experiment_tag": build_experiment_tag(args),
    }


def make_transformer_dataset_class(Dataset: Any) -> Any:
    class ReviewDataset(Dataset):
        def __init__(
            self,
            texts: list[str],
            labels: list[int],
            tokenizer: Any,
            max_length: int,
        ) -> None:
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int) -> dict[str, Any]:
            encoded = self.tokenizer(
                self.texts[index],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = self.labels[index]
            return item

    return ReviewDataset


def train_transformer_candidate(
    model_name: str,
    pretrained_model_name: str,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    args: argparse.Namespace,
    save_output: bool = True,
) -> dict[str, Any]:
    (
        torch,
        DataLoader,
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    ) = require_transformers()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_texts, train_labels = get_texts_and_labels(train_records)
    validation_texts, validation_labels = get_texts_and_labels(validation_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=4,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    ReviewDataset = make_transformer_dataset_class(Dataset)
    train_dataset = ReviewDataset(
        train_texts,
        train_labels,
        tokenizer,
        args.transformer_max_length,
    )
    validation_dataset = ReviewDataset(
        validation_texts,
        validation_labels,
        tokenizer,
        args.transformer_max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.transformer_batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.transformer_batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.transformer_learning_rate,
        weight_decay=args.transformer_weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.transformer_epochs)
    warmup_steps = math.floor(total_steps * args.transformer_warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    class_weight_values = make_class_weights(
        train_labels,
        mode=args.class_weight_mode,
        effective_number_beta=args.effective_number_beta,
        custom_class_weights=args.custom_class_weights,
    )
    class_weights = (
        torch.tensor(class_weight_values, dtype=torch.float, device=device)
        if class_weight_values is not None
        else None
    )
    loss_fn = make_loss_fn(torch, class_weights, args)

    for epoch in range(args.transformer_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())

        average_loss = total_loss / max(1, len(train_loader))
        print(f"{model_name} epoch {epoch + 1}: train_loss={average_loss:.4f}")

    model.eval()
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for batch in validation_loader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=-1)
            y_pred.extend(probabilities.argmax(dim=-1).detach().cpu().tolist())
            y_prob.extend(probabilities.detach().cpu().tolist())

    y_pred = [int(pred) for pred in y_pred]
    y_prob = [[float(value) for value in row] for row in y_prob]
    metrics = evaluate_predictions(validation_labels, y_pred)
    predictions_path = None
    if save_output:
        predictions_path = save_validation_predictions(
            model_name=model_name,
            metrics=metrics,
            validation_records=validation_records,
            y_pred=y_pred,
            args=args,
            y_prob=y_prob,
        )

    return {
        "model_name": model_name,
        "metrics": metrics,
        "validation_predictions_path": str(predictions_path) if predictions_path else None,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "config": {
            "pretrained_model_name": pretrained_model_name,
            "num_labels": 4,
            "epochs": args.transformer_epochs,
            "batch_size": args.transformer_batch_size,
            "learning_rate": args.transformer_learning_rate,
            "max_length": args.transformer_max_length,
            "warmup_ratio": args.transformer_warmup_ratio,
            "weight_decay": args.transformer_weight_decay,
            "loss_config": get_loss_config(train_labels, class_weight_values, args),
            "device": str(device),
        },
    }


def build_fold_prediction_rows(
    fold_index: int,
    validation_records: list[dict[str, Any]],
    y_pred: list[int],
    y_prob: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    rows = build_validation_predictions(validation_records, y_pred, y_prob)
    for row in rows:
        row["fold"] = fold_index
    return rows


def run_tfidf_logreg_kfold(
    folds: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_predictions = []
    fold_metrics = []

    for fold in folds:
        fold_index = int(fold["fold"])
        train_records = fold["train_records"]
        validation_records = fold["validation_records"]
        train_texts, train_labels = get_texts_and_labels(train_records)
        validation_texts, validation_labels = get_texts_and_labels(validation_records)

        y_pred = fit_predict_tfidf_logreg(
            train_texts=train_texts,
            train_labels=train_labels,
            validation_texts=validation_texts,
            args=args,
        )
        metrics = evaluate_predictions(validation_labels, y_pred)
        fold_metrics.append({"fold": fold_index, "metrics": metrics})
        all_predictions.extend(
            build_fold_prediction_rows(fold_index, validation_records, y_pred)
        )
        print(f"tfidf_logreg fold {fold_index} metrics:", metrics)

    aggregate_metrics = summarize_fold_metrics(fold_metrics)
    oof_metrics = evaluate_predictions(
        [int(row["label"]) for row in all_predictions],
        [int(row["pred"]) for row in all_predictions],
    )
    predictions_path = save_kfold_validation_predictions(
        model_name="tfidf_logreg",
        aggregate_metrics=aggregate_metrics,
        fold_metrics=fold_metrics,
        predictions=all_predictions,
        args=args,
        oof_metrics=oof_metrics,
    )

    return {
        "model_name": "tfidf_logreg",
        "metrics": aggregate_metrics,
        "fold_metrics": fold_metrics,
        "validation_predictions_path": str(predictions_path),
        "config": get_tfidf_logreg_config(),
    }


def run_transformer_kfold(
    model_name: str,
    pretrained_model_name: str,
    folds: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_predictions = []
    fold_metrics = []
    config = None

    for fold in folds:
        fold_index = int(fold["fold"])
        result = train_transformer_candidate(
            model_name=model_name,
            pretrained_model_name=pretrained_model_name,
            train_records=fold["train_records"],
            validation_records=fold["validation_records"],
            args=args,
            save_output=False,
        )
        validation_labels = [
            int(record["label"]) for record in fold["validation_records"]
        ]
        y_pred = [int(pred) for pred in result["y_pred"]]
        metrics = evaluate_predictions(validation_labels, y_pred)
        fold_metrics.append({"fold": fold_index, "metrics": metrics})
        y_prob = result.get("y_prob")
        all_predictions.extend(
            build_fold_prediction_rows(
                fold_index,
                fold["validation_records"],
                y_pred,
                y_prob,
            )
        )
        config = result["config"]
        print(f"{model_name} fold {fold_index} metrics:", metrics)

    aggregate_metrics = summarize_fold_metrics(fold_metrics)
    oof_metrics = evaluate_predictions(
        [int(row["label"]) for row in all_predictions],
        [int(row["pred"]) for row in all_predictions],
    )
    predictions_path = save_kfold_validation_predictions(
        model_name=model_name,
        aggregate_metrics=aggregate_metrics,
        fold_metrics=fold_metrics,
        predictions=all_predictions,
        args=args,
        oof_metrics=oof_metrics,
    )

    return {
        "model_name": model_name,
        "metrics": aggregate_metrics,
        "fold_metrics": fold_metrics,
        "validation_predictions_path": str(predictions_path),
        "config": config,
    }


def load_saved_prediction_output(
    model_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = (
        Path(args.output_dir)
        / f"validation_predictions_{append_experiment_tag(model_name, args)}_seed{args.seed}_k{args.cv_folds}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing saved prediction file: {path}")
    return json_read(path)


def json_read(path: str | Path) -> Any:
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def run_saved_prediction_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    outputs = [load_saved_prediction_output(model_name, args) for model_name in args.ensemble_models]
    prediction_maps = [
        {str(item["review_id"]): item for item in output["predictions"]}
        for output in outputs
    ]
    review_ids = sorted(prediction_maps[0])
    if any(set(prediction_map) != set(review_ids) for prediction_map in prediction_maps):
        raise ValueError("All ensemble prediction files must contain the same review_ids.")

    if args.ensemble_weights is None:
        if args.ensemble_method == "weighted_soft":
            weights = [float(output["metrics"].get("qwk_mean", 1.0)) for output in outputs]
        else:
            weights = [1.0 for _ in outputs]
    else:
        weights = args.ensemble_weights

    if len(weights) != len(outputs):
        raise ValueError("--ensemble-weights must match --ensemble-models length.")
    weights = normalize_weights(weights)

    rows = []
    y_true = []
    y_pred = []
    for review_id in review_ids:
        items = [prediction_map[review_id] for prediction_map in prediction_maps]
        label = int(items[0]["label"])
        fold = items[0].get("fold")

        if args.ensemble_method == "hard":
            scores = [0.0, 0.0, 0.0, 0.0]
            for item, weight in zip(items, weights, strict=True):
                scores[int(item["pred"])] += weight
        else:
            if any("probabilities" not in item for item in items):
                raise ValueError(
                    "soft ensemble requires saved probabilities. "
                    "Rerun the selected transformer models with the updated code first."
                )
            scores = [0.0, 0.0, 0.0, 0.0]
            for item, weight in zip(items, weights, strict=True):
                for label_index, probability in enumerate(item["probabilities"]):
                    scores[label_index] += weight * float(probability)

        pred = argmax(scores)
        row = {
            "review_id": items[0]["review_id"],
            "label": label,
            "pred": pred,
            "fold": fold,
            "ensemble_scores": scores,
            "member_predictions": {
                model_name: int(item["pred"])
                for model_name, item in zip(args.ensemble_models, items, strict=True)
            },
        }
        rows.append(row)
        y_true.append(label)
        y_pred.append(pred)

    metrics = evaluate_predictions(y_true, y_pred)
    output = {
        "model_name": "ensemble_" + "_".join(args.ensemble_models),
        "split": f"stratified_kfold_k{args.cv_folds}_seed_{args.seed}",
        "ensemble_method": args.ensemble_method,
        "ensemble_models": args.ensemble_models,
        "ensemble_weights": weights,
        "metrics": metrics,
        "predictions": rows,
    }
    output_path = (
        Path(args.output_dir)
        / f"validation_predictions_ensemble_{args.ensemble_method}_seed{args.seed}_k{args.cv_folds}.json"
    )
    write_json(output, output_path)
    print("ensemble metrics:", metrics)
    print("Saved ensemble predictions:", output_path)
    return output


def run_model_candidates(
    folds: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    comparison_rows = []

    if "tfidf_logreg" in args.models:
        result = run_tfidf_logreg_kfold(
            folds=folds,
            args=args,
        )
        comparison_rows.append(result)
        print("tfidf_logreg metrics:", result["metrics"])

    for model_name, pretrained_model_name in TRANSFORMER_MODEL_CANDIDATES.items():
        if model_name not in args.models:
            continue

        result = run_transformer_kfold(
            model_name=model_name,
            pretrained_model_name=pretrained_model_name,
            folds=folds,
            args=args,
        )
        comparison_rows.append(result)
        print(f"{model_name} metrics:", result["metrics"])

    comparison_path = save_model_comparison(comparison_rows, args)
    print("Saved model comparison:", comparison_path)
    return comparison_rows


def main() -> None:
    args = parse_args()
    records = load_review_data(args.data_path)
    prepared_records = prepare_records(records)
    folds = stratified_kfold_split(
        records=prepared_records,
        n_splits=args.cv_folds,
        seed=args.seed,
    )
    augmented_records = load_augmented_records(args.augmentation_paths)
    folds = append_augmented_train_records(folds, augmented_records)
    save_prepared_outputs(
        prepared_records=prepared_records,
        folds=folds,
        args=args,
    )
    if augmented_records:
        print("Augmented records added to each training fold:", len(augmented_records))
        print("Augmented label distribution:", label_distribution(augmented_records))
    if args.ensemble_only:
        run_saved_prediction_ensemble(args)
    elif not args.prepare_only:
        run_model_candidates(
            folds=folds,
            args=args,
        )


if __name__ == "__main__":
    main()
