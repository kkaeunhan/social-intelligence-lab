from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "task2.2_network" / "data" / "review_keywords.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "task2.2_network" / "data"
DEFAULT_TOP_N = 20
DEFAULT_MIN_DOC_FREQ = 5
DEFAULT_MAX_DOC_RATIO = 0.4
DEFAULT_MIN_PAIR_COUNT = 3
DEFAULT_MIN_NPMI = 0.0
MAX_SAMPLE_REVIEW_IDS = 10
TOP_PAIRS_PER_SCOPE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate keyword co-occurrence, PMI, and NPMI for review keywords."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-doc-freq", type=int, default=DEFAULT_MIN_DOC_FREQ)
    parser.add_argument("--max-doc-ratio", type=float, default=DEFAULT_MAX_DOC_RATIO)
    parser.add_argument("--min-pair-count", type=int, default=DEFAULT_MIN_PAIR_COUNT)
    parser.add_argument("--min-npmi", type=float, default=DEFAULT_MIN_NPMI)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, documents: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def keyword_score(keyword: dict[str, Any]) -> tuple[float, int, str]:
    score = float(keyword.get("count") or 0)
    pos_group = keyword.get("pos_group")
    sources = set(keyword.get("sources") or [])
    if pos_group == "phrase":
        score += 1.25
    elif pos_group == "noun":
        score += 0.5
    elif pos_group in {"adjective", "verb"}:
        score += 0.25
    if "title" in sources:
        score += 0.75
    if "survey_text" in sources:
        score += 0.5
    return (-score, -int(keyword.get("count") or 0), str(keyword.get("normalized") or ""))


def build_global_keyword_df(reviews: list[dict[str, Any]]) -> Counter[str]:
    df: Counter[str] = Counter()
    for review in reviews:
        seen = {
            keyword["normalized"]
            for keyword in review.get("keywords") or []
            if keyword.get("normalized")
        }
        df.update(seen)
    return df


def filter_review_keywords(
    reviews: list[dict[str, Any]],
    *,
    top_n: int,
    min_doc_freq: int,
    max_doc_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    review_count = len(reviews)
    global_df = build_global_keyword_df(reviews)
    max_doc_freq = math.floor(review_count * max_doc_ratio)
    allowed = {
        keyword
        for keyword, df in global_df.items()
        if df >= min_doc_freq and df <= max_doc_freq
    }
    removed_low_df = sum(1 for df in global_df.values() if df < min_doc_freq)
    removed_high_df = sum(1 for df in global_df.values() if df > max_doc_freq)

    network_reviews: list[dict[str, Any]] = []
    total_before_top_n = 0
    total_after_top_n = 0
    for review in reviews:
        candidates = [
            keyword
            for keyword in review.get("keywords") or []
            if keyword.get("normalized") in allowed
        ]
        candidates.sort(key=keyword_score)
        filtered_keywords = candidates[:top_n]
        total_before_top_n += len(candidates)
        total_after_top_n += len(filtered_keywords)
        network_reviews.append(
            {
                "review_id": review["review_id"],
                "product_name": review["product_name"],
                "rating": review["rating"],
                "sentiment_id": review["sentiment_id"],
                "keywords": filtered_keywords,
            }
        )

    filter_summary = {
        "raw_keyword_count": len(global_df),
        "allowed_keyword_count": len(allowed),
        "removed_low_doc_freq_keyword_count": removed_low_df,
        "removed_high_doc_ratio_keyword_count": removed_high_df,
        "max_doc_freq": max_doc_freq,
        "keyword_mentions_before_top_n": total_before_top_n,
        "keyword_mentions_after_top_n": total_after_top_n,
        "empty_review_count": sum(1 for review in network_reviews if not review["keywords"]),
    }
    return network_reviews, filter_summary


def scope_keys(review: dict[str, Any]) -> list[tuple[str, str | None, str | None]]:
    product_name = review["product_name"]
    sentiment_id = review["sentiment_id"]
    return [
        ("global", None, None),
        ("product", product_name, None),
        ("sentiment", None, sentiment_id),
        ("product_sentiment", product_name, sentiment_id),
    ]


def empty_scope_stats() -> dict[str, Any]:
    return {
        "review_count": 0,
        "keyword_doc_counts": Counter(),
        "pair_doc_counts": Counter(),
        "pair_sample_review_ids": defaultdict(list),
    }


def pair_key(keyword_a: dict[str, Any], keyword_b: dict[str, Any]) -> tuple[str, str]:
    normalized_a = keyword_a["normalized"]
    normalized_b = keyword_b["normalized"]
    if normalized_a <= normalized_b:
        return normalized_a, normalized_b
    return normalized_b, normalized_a


def calculate_scope_stats(
    network_reviews: list[dict[str, Any]],
) -> dict[tuple[str, str | None, str | None], dict[str, Any]]:
    scopes: dict[tuple[str, str | None, str | None], dict[str, Any]] = defaultdict(
        empty_scope_stats
    )
    for review in network_reviews:
        unique_keywords = {
            keyword["normalized"]: keyword
            for keyword in review.get("keywords") or []
            if keyword.get("normalized")
        }
        keyword_names = sorted(unique_keywords)
        review_pairs = [
            pair_key(unique_keywords[a], unique_keywords[b])
            for a, b in itertools.combinations(keyword_names, 2)
        ]

        for scope in scope_keys(review):
            stats = scopes[scope]
            stats["review_count"] += 1
            stats["keyword_doc_counts"].update(keyword_names)
            stats["pair_doc_counts"].update(review_pairs)
            for pair in review_pairs:
                samples = stats["pair_sample_review_ids"][pair]
                if len(samples) < MAX_SAMPLE_REVIEW_IDS:
                    samples.append(review["review_id"])
    return scopes


def pmi_npmi(
    *,
    co_count: int,
    keyword_a_count: int,
    keyword_b_count: int,
    review_count: int,
) -> tuple[float, float, float, float]:
    p_a = keyword_a_count / review_count
    p_b = keyword_b_count / review_count
    p_ab = co_count / review_count
    lift = p_ab / (p_a * p_b)
    pmi = math.log2(lift)
    npmi = pmi / -math.log2(p_ab)
    confidence = co_count / keyword_a_count
    return pmi, npmi, confidence, lift


def build_co_occurrence_edges(
    scopes: dict[tuple[str, str | None, str | None], dict[str, Any]],
    *,
    min_pair_count: int,
    min_npmi: float,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for (scope, product_name, sentiment_id), stats in scopes.items():
        review_count = stats["review_count"]
        keyword_doc_counts: Counter[str] = stats["keyword_doc_counts"]
        pair_doc_counts: Counter[tuple[str, str]] = stats["pair_doc_counts"]

        for (keyword_a, keyword_b), co_count in pair_doc_counts.items():
            if co_count < min_pair_count:
                continue
            keyword_a_count = keyword_doc_counts[keyword_a]
            keyword_b_count = keyword_doc_counts[keyword_b]
            pmi, npmi, confidence, lift = pmi_npmi(
                co_count=co_count,
                keyword_a_count=keyword_a_count,
                keyword_b_count=keyword_b_count,
                review_count=review_count,
            )
            if npmi <= min_npmi:
                continue
            edges.append(
                {
                    "source_keyword_id": keyword_id(keyword_a),
                    "target_keyword_id": keyword_id(keyword_b),
                    "keyword_1": keyword_a,
                    "keyword_2": keyword_b,
                    "count": co_count,
                    "keyword_a": keyword_a,
                    "keyword_b": keyword_b,
                    "keyword_a_id": keyword_id(keyword_a),
                    "keyword_b_id": keyword_id(keyword_b),
                    "pair_key": f"{keyword_a}||{keyword_b}",
                    "scope": scope,
                    "product_name": product_name,
                    "sentiment_id": sentiment_id,
                    "co_count": co_count,
                    "keyword_a_count": keyword_a_count,
                    "keyword_b_count": keyword_b_count,
                    "review_count": review_count,
                    "pmi": round(pmi, 6),
                    "npmi": round(npmi, 6),
                    "confidence": round(confidence, 6),
                    "lift": round(lift, 6),
                    "sample_review_ids": stats["pair_sample_review_ids"][
                        (keyword_a, keyword_b)
                    ],
                }
            )

    edges.sort(
        key=lambda edge: (
            edge["scope"],
            edge.get("product_name") or "",
            edge.get("sentiment_id") or "",
            -edge["npmi"],
            -edge["co_count"],
            edge["keyword_a"],
            edge["keyword_b"],
        )
    )
    return edges


def keyword_id(normalized: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z가-힣\s_]+", "", normalized.strip().lower())
    safe = re.sub(r"\s+", "_", safe)
    return f"kw:{safe}"


def summarize_edges(edges: list[dict[str, Any]]) -> dict[str, Any]:
    by_scope: dict[str, Counter[str]] = defaultdict(Counter)
    top_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        scope_label = edge["scope"]
        if edge.get("product_name"):
            scope_label += f":{edge['product_name']}"
        if edge.get("sentiment_id"):
            scope_label += f":{edge['sentiment_id']}"
        by_scope[edge["scope"]]["edge_count"] += 1
        if len(top_by_scope[scope_label]) < TOP_PAIRS_PER_SCOPE:
            top_by_scope[scope_label].append(
                {
                    "keyword_a": edge["keyword_a"],
                    "keyword_b": edge["keyword_b"],
                    "co_count": edge["co_count"],
                    "pmi": edge["pmi"],
                    "npmi": edge["npmi"],
                    "lift": edge["lift"],
                }
            )
    return {
        "edge_counts_by_scope": {
            scope: dict(counter) for scope, counter in sorted(by_scope.items())
        },
        "top_pairs_by_scope": dict(sorted(top_by_scope.items())),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reviews = load_jsonl(args.input)
    network_reviews, filter_summary = filter_review_keywords(
        reviews,
        top_n=args.top_n,
        min_doc_freq=args.min_doc_freq,
        max_doc_ratio=args.max_doc_ratio,
    )
    scopes = calculate_scope_stats(network_reviews)
    edges = build_co_occurrence_edges(
        scopes,
        min_pair_count=args.min_pair_count,
        min_npmi=args.min_npmi,
    )

    output_network_reviews = args.output_dir / "network_review_keywords.jsonl"
    output_edges = args.output_dir / "co_occurrence_edges.jsonl"
    output_summary = args.output_dir / "pmi_npmi_summary.json"

    write_jsonl(output_network_reviews, network_reviews)
    write_jsonl(output_edges, edges)

    keywords_per_review = [len(review["keywords"]) for review in network_reviews]
    positive_npmi_edges = [edge for edge in edges if edge["npmi"] > 0]
    summary = {
        "input_path": str(args.input),
        "output_network_reviews": str(output_network_reviews),
        "output_edges": str(output_edges),
        "review_count": len(reviews),
        "top_n": args.top_n,
        "min_doc_freq": args.min_doc_freq,
        "max_doc_ratio": args.max_doc_ratio,
        "min_pair_count": args.min_pair_count,
        "min_npmi": args.min_npmi,
        "filter_summary": filter_summary,
        "keywords_per_review_after_filter": {
            "min": min(keywords_per_review, default=0),
            "max": max(keywords_per_review, default=0),
            "avg": round(sum(keywords_per_review) / len(keywords_per_review), 2)
            if keywords_per_review
            else 0,
        },
        "co_occurrence_edge_count": len(edges),
        "positive_npmi_edge_count": len(positive_npmi_edges),
        **summarize_edges(edges),
    }
    with output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
