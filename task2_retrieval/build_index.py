import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from elastic_search import (  # noqa: E402
    DEFAULT_INDEX_MAPPING,
    create_index,
    create_openai_enricher,
    enrich_document_with_llm,
    get_client,
    helpers,
    load_config,
    preprocess_record,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "task2_retrieval"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "indexed_documents.jsonl"
DEFAULT_FAILED_LOG = DEFAULT_OUTPUT_DIR / "index_failures.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Build and index retrieval documents.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT_DIR / "si_dataset" / "review_for_analysis.json",
        help="Path to raw review JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N raw records. Useful for smoke tests.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable OpenAI metadata enrichment before indexing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of processed documents to checkpoint and index per batch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="JSONL path for successfully processed documents.",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=DEFAULT_FAILED_LOG,
        help="JSONL path for records that failed preprocessing/enrichment/indexing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip review_ids already present in the checkpoint.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Only build checkpoint documents. Do not send batches to Elasticsearch.",
    )
    parser.add_argument(
        "--index-from-checkpoint",
        action="store_true",
        help="Index documents already stored in checkpoint instead of rebuilding them.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Elasticsearch index before indexing.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Override ELASTICSEARCH_INDEX from .env.",
    )
    return parser.parse_args()


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_checkpoint_ids(path: Path) -> set[str]:
    review_ids = set()
    if not path.exists():
        return review_ids

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue
            review_id = str(document.get("review_id") or "").strip()
            if review_id:
                review_ids.add(review_id)
    return review_ids


def iter_checkpoint_documents(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def content_fingerprint(content: str) -> str:
    normalized = content.lower()
    normalized = "".join(normalized.split())
    return "".join(ch for ch in normalized if ch.isalnum() or "가" <= ch <= "힣")


def index_batch(client, index_name: str, documents: list[dict[str, Any]]) -> tuple[int, list[Any]]:
    if helpers is None:
        raise ImportError("Install elasticsearch first: pip install elasticsearch")
    if not documents:
        return 0, []

    actions = [
        {
            "_index": index_name,
            "_id": document["review_id"],
            "_source": document,
        }
        for document in documents
    ]
    return helpers.bulk(client, actions, refresh=False, raise_on_error=False)


def flush_batch(
    batch: list[dict[str, Any]],
    *,
    checkpoint_path: Path,
    client,
    index_name: str,
    no_index: bool,
    failed_log: Path,
) -> tuple[int, int]:
    if not batch:
        return 0, 0

    for document in batch:
        append_jsonl(checkpoint_path, document)

    if no_index:
        return len(batch), 0

    indexed_count, errors = index_batch(client, index_name, batch)
    for error in errors:
        append_jsonl(failed_log, {"stage": "index", "error": error})
    return indexed_count, len(errors)


def build_and_index(args, records, client, index_name: str) -> None:
    llm_enricher = create_openai_enricher() if args.llm else None
    checkpoint_ids = load_checkpoint_ids(args.checkpoint) if args.resume else set()
    seen_review_ids = set(checkpoint_ids)
    seen_contents = load_checkpoint_content_fingerprints(args.checkpoint) if args.resume else set()
    batch: list[dict[str, Any]] = []
    kept = 0
    indexed_total = 0
    index_errors = 0
    skipped_resume = 0
    failed_total = 0

    for i, record in enumerate(records, start=1):
        raw_review_id = str(record.get("reviewId") or record.get("review_id") or "").strip()
        if raw_review_id and raw_review_id in checkpoint_ids:
            skipped_resume += 1
            if i % args.batch_size == 0 or i == len(records):
                print_progress(i, len(records), kept, indexed_total, failed_total, skipped_resume)
            continue

        try:
            document = preprocess_record(record)
        except Exception as exc:
            failed_total += 1
            append_jsonl(
                args.failed_log,
                {
                    "stage": "preprocess",
                    "review_id": raw_review_id,
                    "error": repr(exc),
                },
            )
            continue

        if document is None:
            continue

        review_id = document["review_id"]
        content_key = content_fingerprint(document["content"])
        if review_id in seen_review_ids or content_key in seen_contents:
            continue

        if llm_enricher is not None:
            try:
                document = enrich_document_with_llm(document, llm_enricher)
            except Exception as exc:
                failed_total += 1
                append_jsonl(
                    args.failed_log,
                    {
                        "stage": "llm_enrich",
                        "review_id": review_id,
                        "error": repr(exc),
                    },
                )
                continue

        seen_review_ids.add(review_id)
        seen_contents.add(content_key)
        batch.append(document)
        kept += 1

        if len(batch) >= args.batch_size:
            indexed_count, error_count = flush_batch(
                batch,
                checkpoint_path=args.checkpoint,
                client=client,
                index_name=index_name,
                no_index=args.no_index,
                failed_log=args.failed_log,
            )
            indexed_total += indexed_count
            index_errors += error_count
            batch.clear()
            print_progress(i, len(records), kept, indexed_total, failed_total, skipped_resume)

    indexed_count, error_count = flush_batch(
        batch,
        checkpoint_path=args.checkpoint,
        client=client,
        index_name=index_name,
        no_index=args.no_index,
        failed_log=args.failed_log,
    )
    indexed_total += indexed_count
    index_errors += error_count

    print_progress(len(records), len(records), kept, indexed_total, failed_total, skipped_resume)
    print(f"checkpoint: {args.checkpoint}")
    print(f"failed log: {args.failed_log}")
    print(f"index errors: {index_errors}")


def print_progress(
    processed: int,
    total: int,
    kept: int,
    indexed: int,
    failed: int,
    skipped_resume: int,
) -> None:
    print(
        f"processed {processed}/{total} | kept {kept} | "
        f"indexed {indexed} | failed {failed} | resumed {skipped_resume}"
    )


def load_checkpoint_content_fingerprints(path: Path) -> set[str]:
    fingerprints = set()
    if not path.exists():
        return fingerprints

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = str(document.get("content") or "")
            if content:
                fingerprints.add(content_fingerprint(content))
    return fingerprints


def index_from_checkpoint(args, client, index_name: str) -> None:
    batch = []
    indexed_total = 0
    index_errors = 0

    for document in iter_checkpoint_documents(args.checkpoint):
        batch.append(document)
        if len(batch) >= args.batch_size:
            indexed_count, errors = index_batch(client, index_name, batch)
            indexed_total += indexed_count
            index_errors += len(errors)
            for error in errors:
                append_jsonl(args.failed_log, {"stage": "index", "error": error})
            batch.clear()
            print(f"indexed from checkpoint: {indexed_total}")

    if batch:
        indexed_count, errors = index_batch(client, index_name, batch)
        indexed_total += indexed_count
        index_errors += len(errors)
        for error in errors:
            append_jsonl(args.failed_log, {"stage": "index", "error": error})

    client.indices.refresh(index=index_name)
    print(f"indexed from checkpoint: {indexed_total}")
    print(f"index errors: {index_errors}")
    print(client.count(index=index_name))


def main():
    args = parse_args()
    config = load_config()
    if args.index_name is not None:
        config.index_name = args.index_name

    with args.input.open(encoding="utf-8") as f:
        records = json.load(f)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"input records: {len(records)}")
    print(f"llm enrichment: {args.llm}")
    print(f"index name: {config.index_name}")
    print(f"recreate index: {args.recreate}")
    print(f"batch size: {args.batch_size}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"failed log: {args.failed_log}")

    client = None if args.no_index else get_client(config)
    if client is not None:
        create_index(
            client,
            index_name=config.index_name,
            mapping=DEFAULT_INDEX_MAPPING,
            recreate=args.recreate,
        )

    if args.index_from_checkpoint:
        if client is None:
            raise ValueError("--index-from-checkpoint cannot be used with --no-index")
        index_from_checkpoint(args, client, config.index_name)
        return

    build_and_index(args, records, client, config.index_name)

    if client is not None:
        client.indices.refresh(index=config.index_name)
        print(client.count(index=config.index_name))


if __name__ == "__main__":
    main()
