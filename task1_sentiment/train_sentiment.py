"""Train and save the final Task 1 sentiment model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task1_sentiment.model_candidates import (
    TRANSFORMER_MODEL_CANDIDATES,
    get_loss_config,
    get_texts_and_labels,
    load_augmented_records,
    make_class_weights,
    make_loss_fn,
    make_transformer_dataset_class,
    prepare_records,
    require_transformers,
)
from sentiment import load_review_data
from utils import write_json


DEFAULT_AUGMENTATION_PATHS = [
    "output/task1_sentiment/llm_augmentation/label0_solar-pro3_target120_batch30_seed42_combined_generated_normalized.json",
    "output/task1_sentiment/llm_augmentation/label1_solar-pro3_target180_batch30_seed42_combined_generated_normalized.json",
    "output/task1_sentiment/llm_augmentation/label2_solar-pro3_target80_batch30_seed42_combined_generated_normalized.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final Task 1 sentiment model.")
    parser.add_argument("--data-path", default="si_dataset/train_review_data.json")
    parser.add_argument(
        "--augmentation-paths",
        nargs="*",
        default=DEFAULT_AUGMENTATION_PATHS,
        help="Generated review files to append to final training data.",
    )
    parser.add_argument(
        "--model",
        default="beomi_kc_electra",
        choices=sorted(TRANSFORMER_MODEL_CANDIDATES),
        help="Model key from task1_sentiment/model_candidates.py.",
    )
    parser.add_argument("--output-dir", default="artifacts/task1_sentiment/final_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--class-weight-mode",
        choices=["inverse", "none", "sqrt_inverse", "effective_number", "custom"],
        default="sqrt_inverse",
    )
    parser.add_argument("--effective-number-beta", type=float, default=0.999)
    parser.add_argument("--custom-class-weights", nargs=4, type=float, default=None)
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "focal"],
        default="focal",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    return parser.parse_args()


def label_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(record["label"]) for record in records)
    return {str(label): counts.get(label, 0) for label in range(4)}


def main() -> None:
    args = parse_args()
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

    original_records = prepare_records(load_review_data(args.data_path))
    augmented_records = load_augmented_records(args.augmentation_paths)
    train_records = original_records + augmented_records
    train_texts, train_labels = get_texts_and_labels(train_records)

    pretrained_model_name = TRANSFORMER_MODEL_CANDIDATES[args.model]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=4,
        ignore_mismatched_sizes=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ReviewDataset = make_transformer_dataset_class(Dataset)
    train_dataset = ReviewDataset(
        train_texts,
        train_labels,
        tokenizer,
        args.max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
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

    for epoch in range(args.epochs):
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
        print(f"{args.model} final epoch {epoch + 1}: train_loss={average_loss:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = {
        "model_key": args.model,
        "pretrained_model_name": pretrained_model_name,
        "num_labels": 4,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "original_count": len(original_records),
        "augmented_count": len(augmented_records),
        "train_count": len(train_records),
        "original_label_distribution": label_distribution(original_records),
        "augmented_label_distribution": label_distribution(augmented_records),
        "train_label_distribution": label_distribution(train_records),
        "augmentation_paths": [str(path) for path in args.augmentation_paths],
        "loss_config": get_loss_config(train_labels, class_weight_values, args),
    }
    write_json(config, output_dir / "training_config.json")
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "0": "negative",
                "1": "weak_negative",
                "2": "weak_positive",
                "3": "positive",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved final model:", output_dir)
    print("Train label distribution:", config["train_label_distribution"])


if __name__ == "__main__":
    main()
