import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from elastic_search import (  # noqa: E402
    BM25_TOP_K,
    DENSE_TOP_K,
    RRF_CANDIDATE_TOP_K,
    RRF_K,
    _search_bm25,
    _search_dense,
    expand_query,
    get_client,
    load_config,
    post_filter_candidates,
    postprocess_results,
    reciprocal_rank_fusion,
    rerank_candidates,
    validate_search_output,
)


DEFAULT_EVAL_PATH = ROOT_DIR / "task2_retrieval" / "eval" / "human_eval.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "task2_retrieval" / "eval"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval with MAP@K.")
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="Path to human_eval.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Override ELASTICSEARCH_INDEX from .env.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N queries.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Final evaluation depth.",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Use RRF ranking as final ranking without cross-encoder reranking.",
    )
    return parser.parse_args()


def load_eval_set(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def review_id_from_item(item: dict[str, Any]) -> str:
    return str(item.get("reviewId") or item.get("review_id") or "").strip()


def gold_review_ids(entry: dict[str, Any]) -> list[str]:
    items = entry.get("reviews_to_evaluate", [])
    return [review_id for item in items if (review_id := review_id_from_item(item))]


def result_ids(results: list[dict[str, Any]], key: str = "review_id") -> list[str]:
    ids = []
    for result in results:
        review_id = str(result.get(key) or result.get("id") or "").strip()
        if review_id:
            ids.append(review_id)
    return ids


def recall_at(candidates: list[str], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return len(set(candidates) & gold) / len(gold)


def average_precision_at(candidates: list[str], gold_ids: list[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen = set()
    for rank, review_id in enumerate(candidates[:k], start=1):
        if review_id in seen:
            continue
        seen.add(review_id)
        if review_id in gold:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(gold), k)


def stage_summary(ids: list[str], gold_ids: list[str], k: int) -> dict[str, Any]:
    top_ids = ids[:k]
    return {
        "ids": top_ids,
        "hits": len(set(top_ids) & set(gold_ids)),
        "recall": recall_at(top_ids, gold_ids),
        "ap": average_precision_at(top_ids, gold_ids, k),
    }


def evaluate_query(client, index_name: str, query: str, gold_ids: list[str], k: int, skip_rerank: bool):
    expansion = expand_query(query)
    bm25_results = _search_bm25(client, index_name, query, size=BM25_TOP_K)

    dense_by_id = {}
    for dense_query in expansion.get("dense_queries", [query]):
        dense_results_for_query = _search_dense(
            client,
            index_name,
            str(dense_query),
            size=DENSE_TOP_K,
            expansion=expansion,
        )
        for result in dense_results_for_query:
            review_id = result.get("review_id")
            if not review_id:
                continue
            current = dense_by_id.get(review_id)
            if current is None or result.get("score", 0.0) > current.get("score", 0.0):
                dense_by_id[review_id] = result
    dense_results = sorted(
        dense_by_id.values(),
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )[:DENSE_TOP_K]

    rrf_candidates = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        k=RRF_K,
        size=RRF_CANDIDATE_TOP_K,
    )

    if skip_rerank:
        final_candidates = [dict(candidate, final_score=candidate.get("rrf_score", 0.0)) for candidate in rrf_candidates]
    else:
        final_candidates = rerank_candidates(query=query, candidates=rrf_candidates)
    final_candidates = post_filter_candidates(final_candidates, size=k, expansion=expansion)
    output = postprocess_results(final_candidates, query=query)
    if not validate_search_output(output):
        raise ValueError(f"Invalid SearchOutput for query: {query}")

    bm25_ids = result_ids(bm25_results)
    dense_ids = result_ids(dense_results)
    rrf_ids = result_ids(rrf_candidates)
    final_ids = [item["review_id"] for item in output["results"]]

    return {
        "bm25": stage_summary(bm25_ids, gold_ids, BM25_TOP_K),
        "dense": stage_summary(dense_ids, gold_ids, DENSE_TOP_K),
        "rrf": stage_summary(rrf_ids, gold_ids, RRF_CANDIDATE_TOP_K),
        "final": stage_summary(final_ids, gold_ids, k),
        "final_results": output["results"],
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    args = parse_args()
    config = load_config()
    if args.index_name is not None:
        config.index_name = args.index_name

    eval_data = load_eval_set(args.eval_path)
    items = list(eval_data.items())
    if args.limit is not None:
        items = items[: args.limit]

    client = get_client(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    details = {}

    for query_id, entry in items:
        query = entry["query"]
        category = entry.get("category", "")
        gold_ids = gold_review_ids(entry)
        metrics = evaluate_query(
            client=client,
            index_name=config.index_name,
            query=query,
            gold_ids=gold_ids,
            k=args.k,
            skip_rerank=args.skip_rerank,
        )

        row = {
            "query_id": query_id,
            "category": category,
            "query": query,
            "gold_count": len(gold_ids),
            "bm25_recall_at_100": metrics["bm25"]["recall"],
            "dense_recall_at_100": metrics["dense"]["recall"],
            "rrf_recall_at_50": metrics["rrf"]["recall"],
            "final_hits_at_k": metrics["final"]["hits"],
            "final_recall_at_k": metrics["final"]["recall"],
            "final_ap_at_k": metrics["final"]["ap"],
            "final_ids": " ".join(metrics["final"]["ids"]),
            "missing_gold_ids": " ".join(sorted(set(gold_ids) - set(metrics["final"]["ids"]))),
        }
        rows.append(row)
        details[query_id] = {
            "query": query,
            "category": category,
            "gold_ids": gold_ids,
            **metrics,
        }
        print(
            f"{query_id} | {category} | AP@{args.k}={row['final_ap_at_k']:.4f} | "
            f"hits={row['final_hits_at_k']}/{min(len(gold_ids), args.k)} | "
            f"BM25 R@100={row['bm25_recall_at_100']:.2f} | "
            f"Dense R@100={row['dense_recall_at_100']:.2f} | "
            f"RRF R@50={row['rrf_recall_at_50']:.2f}"
        )

    summary = {
        "query_count": len(rows),
        "map_at_k": mean([row["final_ap_at_k"] for row in rows]),
        "mean_final_recall_at_k": mean([row["final_recall_at_k"] for row in rows]),
        "mean_bm25_recall_at_100": mean([row["bm25_recall_at_100"] for row in rows]),
        "mean_dense_recall_at_100": mean([row["dense_recall_at_100"] for row in rows]),
        "mean_rrf_recall_at_50": mean([row["rrf_recall_at_50"] for row in rows]),
        "by_category": {},
    }

    categories = sorted({row["category"] for row in rows})
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        summary["by_category"][category] = {
            "query_count": len(category_rows),
            "map_at_k": mean([row["final_ap_at_k"] for row in category_rows]),
            "mean_final_recall_at_k": mean([row["final_recall_at_k"] for row in category_rows]),
        }

    csv_path = args.output_dir / "retrieval_eval.csv"
    json_path = args.output_dir / "retrieval_eval_details.json"
    summary_path = args.output_dir / "retrieval_eval_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved details: {json_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
