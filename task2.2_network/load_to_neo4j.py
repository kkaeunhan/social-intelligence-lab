from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT_DIR / "task2.2_network"
DATA_DIR = TASK_DIR / "data"

DEFAULT_CLEANED_REVIEWS = DATA_DIR / "cleaned_reviews.jsonl"
DEFAULT_KEYWORDS = DATA_DIR / "keywords.jsonl"
DEFAULT_REVIEW_KEYWORD_EDGES = DATA_DIR / "review_keyword_edges.jsonl"
DEFAULT_CO_OCCURRENCE_EDGES = DATA_DIR / "co_occurrence_edges.jsonl"
DEFAULT_SCHEMA = TASK_DIR / "cypher" / "schema.cypher"
DEFAULT_SUMMARY = DATA_DIR / "neo4j_load_summary.json"

SENTIMENTS = [
    {
        "sentiment_id": "positive",
        "label": "긍정",
        "rating_min": 4,
        "rating_max": 5,
        "rule": "rating 4-5",
    },
    {
        "sentiment_id": "neutral",
        "label": "중립",
        "rating_min": 3,
        "rating_max": 3,
        "rule": "rating 3",
    },
    {
        "sentiment_id": "negative",
        "label": "부정",
        "rating_min": 1,
        "rating_max": 2,
        "rule": "rating 1-2",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load task2.2 network data into Neo4j.")
    parser.add_argument("--cleaned-reviews", type=Path, default=DEFAULT_CLEANED_REVIEWS)
    parser.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS)
    parser.add_argument("--review-keyword-edges", type=Path, default=DEFAULT_REVIEW_KEYWORD_EDGES)
    parser.add_argument("--co-occurrence-edges", type=Path, default=DEFAULT_CO_OCCURRENCE_EDGES)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip schema.cypher execution.",
    )
    parser.add_argument(
        "--skip-co-occurrence",
        action="store_true",
        help="Skip CO_OCCURS_WITH loading. Useful for smoke tests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Load only the first N rows from each data file for smoke testing.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete task2.2 Product/Review/Keyword/Sentiment nodes and relationships before loading.",
    )
    return parser.parse_args()


def load_neo4j_config() -> dict[str, str]:
    load_dotenv(ROOT_DIR / ".env")
    required = {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing Neo4j env vars: {', '.join(missing)}")
    return {key: str(value) for key, value in required.items()}


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            if limit is not None and index > limit:
                break
            if line.strip():
                yield json.loads(line)


def batched(iterable: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_write_batches(
    driver,
    database: str,
    query: str,
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int,
    label: str,
) -> int:
    total = 0
    with driver.session(database=database) as session:
        for batch in batched(rows, batch_size):
            session.execute_write(lambda tx: tx.run(query, rows=batch).consume())
            total += len(batch)
            print(f"{label}: loaded {total}")
    return total


def split_cypher_statements(text: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(current).strip()
            statements.append(statement.rstrip(";").strip())
            current = []
    if current:
        statements.append("\n".join(current).strip())
    return [statement for statement in statements if statement]


def run_schema(driver, database: str, schema_path: Path) -> int:
    statements = split_cypher_statements(schema_path.read_text(encoding="utf-8"))
    with driver.session(database=database) as session:
        for statement in statements:
            session.execute_write(lambda tx, q=statement: tx.run(q).consume())
    return len(statements)


def reset_task_graph(driver, database: str) -> None:
    statements = [
        "MATCH ()-[r:HAS_REVIEW|HAS_SENTIMENT|MENTIONS|CO_OCCURS_WITH]->() DELETE r",
        "MATCH (n) WHERE n:Product OR n:Review OR n:Keyword OR n:Sentiment DETACH DELETE n",
    ]
    with driver.session(database=database) as session:
        for statement in statements:
            session.execute_write(lambda tx, q=statement: tx.run(q).consume())


def product_family(product_name: str) -> str:
    if product_name.startswith("iphone"):
        return "iphone"
    if product_name.startswith("galaxy_s"):
        return "galaxy_s"
    if product_name.startswith("galaxy_z"):
        return "galaxy_z"
    return "unknown"


def brand(product_name: str) -> str:
    if product_name.startswith("iphone"):
        return "Apple"
    if product_name.startswith("galaxy"):
        return "Samsung"
    return "unknown"


def product_rows_from_reviews(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    for review in iter_jsonl(path, limit):
        product_name = review["product_name"]
        products.setdefault(
            product_name,
            {
                "product_name": product_name,
                "display_name": product_name,
                "product_family": product_family(product_name),
                "brand": brand(product_name),
                "review_count": 0,
            },
        )
        products[product_name]["review_count"] += 1
    return sorted(products.values(), key=lambda row: row["product_name"])


def review_rows(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    for review in iter_jsonl(path, limit):
        yield {
            "review_id": review["review_id"],
            "source_review_id": review.get("source_review_id"),
            "product_name": review["product_name"],
            "product_id": review.get("product_id"),
            "vendor_item_id": review.get("vendor_item_id"),
            "item_id": review.get("item_id"),
            "item_name": review.get("item_name"),
            "title": review.get("title"),
            "content": review.get("content"),
            "survey_text": review.get("survey_text"),
            "analysis_text": review.get("analysis_text"),
            "analysis_char_len": review.get("analysis_char_len"),
            "analysis_meaningful_chars": review.get("analysis_meaningful_chars"),
            "rating": review.get("rating"),
            "sentiment_id": review.get("sentiment_id"),
            "review_at": review.get("review_at"),
            "created_at": review.get("created_at"),
            "helpful_count": review.get("helpful_count"),
            "helpful_true_count": review.get("helpful_true_count"),
            "helpful_false_count": review.get("helpful_false_count"),
            "has_image": review.get("has_image"),
            "has_video": review.get("has_video"),
        }


def keyword_rows(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    for keyword in iter_jsonl(path, limit):
        yield {
            "keyword_id": keyword["keyword_id"],
            "text": keyword.get("text"),
            "normalized": keyword.get("normalized"),
            "pos_group": keyword.get("pos_group"),
            "df": keyword.get("df"),
            "tf": keyword.get("tf"),
            "product_df_json": json.dumps(keyword.get("product_df") or {}, ensure_ascii=False),
            "sentiment_df_json": json.dumps(keyword.get("sentiment_df") or {}, ensure_ascii=False),
            "is_stopword": keyword.get("is_stopword", False),
        }


def mention_rows(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    for edge in iter_jsonl(path, limit):
        yield {
            "review_id": edge["review_id"],
            "keyword_id": edge["keyword_id"],
            "product_name": edge.get("product_name"),
            "sentiment_id": edge.get("sentiment_id"),
            "count": edge.get("count"),
            "sources": edge.get("sources") or [],
            "weight": edge.get("weight", 1.0),
            "pos_group": edge.get("pos_group"),
        }


def co_occurrence_rows(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    for edge in iter_jsonl(path, limit):
        source_keyword_id = edge.get("source_keyword_id") or edge.get("keyword_a_id")
        target_keyword_id = edge.get("target_keyword_id") or edge.get("keyword_b_id")
        scope = edge.get("scope")
        product_name = edge.get("product_name")
        sentiment_id = edge.get("sentiment_id")
        pair_key = edge.get("pair_key") or f"{edge.get('keyword_1')}||{edge.get('keyword_2')}"
        rel_key = "|".join(
            [
                str(scope or ""),
                str(product_name or ""),
                str(sentiment_id or ""),
                str(pair_key),
            ]
        )
        yield {
            "source_keyword_id": source_keyword_id,
            "target_keyword_id": target_keyword_id,
            "keyword_1": edge.get("keyword_1") or edge.get("keyword_a"),
            "keyword_2": edge.get("keyword_2") or edge.get("keyword_b"),
            "pair_key": pair_key,
            "rel_key": rel_key,
            "scope": scope,
            "product_name": product_name,
            "sentiment_id": sentiment_id,
            "count": edge.get("count") or edge.get("co_count"),
            "co_count": edge.get("co_count") or edge.get("count"),
            "keyword_a_count": edge.get("keyword_a_count"),
            "keyword_b_count": edge.get("keyword_b_count"),
            "review_count": edge.get("review_count"),
            "pmi": edge.get("pmi"),
            "npmi": edge.get("npmi"),
            "confidence": edge.get("confidence"),
            "lift": edge.get("lift"),
            "sample_review_ids": edge.get("sample_review_ids") or [],
        }


UPSERT_PRODUCTS = """
UNWIND $rows AS row
MERGE (p:Product {product_name: row.product_name})
SET p.display_name = row.display_name,
    p.product_family = row.product_family,
    p.brand = row.brand,
    p.review_count = row.review_count
"""

UPSERT_SENTIMENTS = """
UNWIND $rows AS row
MERGE (s:Sentiment {sentiment_id: row.sentiment_id})
SET s.label = row.label,
    s.rating_min = row.rating_min,
    s.rating_max = row.rating_max,
    s.rule = row.rule
"""

UPSERT_REVIEWS = """
UNWIND $rows AS row
MERGE (r:Review {review_id: row.review_id})
SET r.source_review_id = row.source_review_id,
    r.product_name = row.product_name,
    r.product_id = row.product_id,
    r.vendor_item_id = row.vendor_item_id,
    r.item_id = row.item_id,
    r.item_name = row.item_name,
    r.title = row.title,
    r.content = row.content,
    r.survey_text = row.survey_text,
    r.analysis_text = row.analysis_text,
    r.analysis_char_len = row.analysis_char_len,
    r.analysis_meaningful_chars = row.analysis_meaningful_chars,
    r.rating = row.rating,
    r.sentiment_id = row.sentiment_id,
    r.review_at = row.review_at,
    r.created_at = row.created_at,
    r.helpful_count = row.helpful_count,
    r.helpful_true_count = row.helpful_true_count,
    r.helpful_false_count = row.helpful_false_count,
    r.has_image = row.has_image,
    r.has_video = row.has_video
WITH row, r
MATCH (p:Product {product_name: row.product_name})
MERGE (p)-[hr:HAS_REVIEW]->(r)
SET hr.review_at = row.review_at,
    hr.rating = row.rating
WITH row, r
MATCH (s:Sentiment {sentiment_id: row.sentiment_id})
MERGE (r)-[hs:HAS_SENTIMENT]->(s)
SET hs.rating = row.rating,
    hs.rule_version = "rating_v1"
"""

UPSERT_KEYWORDS = """
UNWIND $rows AS row
MERGE (k:Keyword {keyword_id: row.keyword_id})
SET k.text = row.text,
    k.normalized = row.normalized,
    k.pos_group = row.pos_group,
    k.df = row.df,
    k.tf = row.tf,
    k.product_df_json = row.product_df_json,
    k.sentiment_df_json = row.sentiment_df_json,
    k.is_stopword = row.is_stopword
"""

UPSERT_MENTIONS = """
UNWIND $rows AS row
MATCH (r:Review {review_id: row.review_id})
MATCH (k:Keyword {keyword_id: row.keyword_id})
MERGE (r)-[m:MENTIONS]->(k)
SET m.count = row.count,
    m.sources = row.sources,
    m.weight = row.weight,
    m.product_name = row.product_name,
    m.sentiment_id = row.sentiment_id,
    m.pos_group = row.pos_group
"""

UPSERT_CO_OCCURRENCES = """
UNWIND $rows AS row
MATCH (a:Keyword {keyword_id: row.source_keyword_id})
MATCH (b:Keyword {keyword_id: row.target_keyword_id})
MERGE (a)-[r:CO_OCCURS_WITH {rel_key: row.rel_key}]->(b)
SET r.source_keyword_id = row.source_keyword_id,
    r.target_keyword_id = row.target_keyword_id,
    r.keyword_1 = row.keyword_1,
    r.keyword_2 = row.keyword_2,
    r.pair_key = row.pair_key,
    r.scope = row.scope,
    r.product_name = row.product_name,
    r.sentiment_id = row.sentiment_id,
    r.count = row.count,
    r.co_count = row.co_count,
    r.keyword_a_count = row.keyword_a_count,
    r.keyword_b_count = row.keyword_b_count,
    r.review_count = row.review_count,
    r.pmi = row.pmi,
    r.npmi = row.npmi,
    r.confidence = row.confidence,
    r.lift = row.lift,
    r.sample_review_ids = row.sample_review_ids
"""

VERIFY_COUNTS = """
CALL () {
  MATCH (n:Product) RETURN "Product" AS name, count(n) AS count
  UNION ALL
  MATCH (n:Review) RETURN "Review" AS name, count(n) AS count
  UNION ALL
  MATCH (n:Keyword) RETURN "Keyword" AS name, count(n) AS count
  UNION ALL
  MATCH (n:Sentiment) RETURN "Sentiment" AS name, count(n) AS count
  UNION ALL
  MATCH ()-[r:HAS_REVIEW]->() RETURN "HAS_REVIEW" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:HAS_SENTIMENT]->() RETURN "HAS_SENTIMENT" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:MENTIONS]->() RETURN "MENTIONS" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:CO_OCCURS_WITH]->() RETURN "CO_OCCURS_WITH" AS name, count(r) AS count
}
RETURN name, count
ORDER BY name
"""

VERIFY_PRODUCT_REVIEWS = """
MATCH (p:Product)-[:HAS_REVIEW]->(r:Review)
RETURN p.product_name AS product_name, count(r) AS review_count
ORDER BY product_name
"""

VERIFY_TOP_GLOBAL = """
MATCH (a:Keyword)-[r:CO_OCCURS_WITH {scope: "global"}]->(b:Keyword)
RETURN a.text AS keyword_1, b.text AS keyword_2, r.count AS count, r.pmi AS pmi, r.npmi AS npmi
ORDER BY r.npmi DESC, r.count DESC
LIMIT 10
"""


def verify(driver, database: str) -> dict[str, Any]:
    with driver.session(database=database) as session:
        counts = {
            record["name"]: record["count"]
            for record in session.run(VERIFY_COUNTS)
        }
        product_review_counts = [dict(record) for record in session.run(VERIFY_PRODUCT_REVIEWS)]
        top_global_pairs = [dict(record) for record in session.run(VERIFY_TOP_GLOBAL)]
    return {
        "counts": counts,
        "product_review_counts": product_review_counts,
        "top_global_pairs": top_global_pairs,
    }


def main() -> None:
    args = parse_args()
    started_at = time.time()
    config = load_neo4j_config()
    summary: dict[str, Any] = {
        "database": config["database"],
        "batch_size": args.batch_size,
        "limit": args.limit,
        "input_files": {
            "cleaned_reviews": str(args.cleaned_reviews),
            "keywords": str(args.keywords),
            "review_keyword_edges": str(args.review_keyword_edges),
            "co_occurrence_edges": str(args.co_occurrence_edges),
            "schema": str(args.schema),
        },
    }

    driver = GraphDatabase.driver(
        config["uri"],
        auth=(config["username"], config["password"]),
    )
    try:
        driver.verify_connectivity()
        if not args.skip_schema:
            schema_statements = run_schema(driver, config["database"], args.schema)
            summary["schema_statements_executed"] = schema_statements
            print(f"schema: executed {schema_statements} statements")
        else:
            summary["schema_statements_executed"] = 0

        if args.reset:
            reset_task_graph(driver, config["database"])
            summary["reset"] = True
            print("reset: deleted task2.2 graph nodes and relationships")
        else:
            summary["reset"] = False

        products = product_rows_from_reviews(args.cleaned_reviews, args.limit)
        summary["loaded_products"] = run_write_batches(
            driver,
            config["database"],
            UPSERT_PRODUCTS,
            products,
            batch_size=args.batch_size,
            label="Product",
        )
        summary["loaded_sentiments"] = run_write_batches(
            driver,
            config["database"],
            UPSERT_SENTIMENTS,
            SENTIMENTS,
            batch_size=args.batch_size,
            label="Sentiment",
        )
        summary["loaded_reviews"] = run_write_batches(
            driver,
            config["database"],
            UPSERT_REVIEWS,
            review_rows(args.cleaned_reviews, args.limit),
            batch_size=args.batch_size,
            label="Review/HAS_REVIEW/HAS_SENTIMENT",
        )
        summary["loaded_keywords"] = run_write_batches(
            driver,
            config["database"],
            UPSERT_KEYWORDS,
            keyword_rows(args.keywords, args.limit),
            batch_size=args.batch_size,
            label="Keyword",
        )
        summary["loaded_mentions"] = run_write_batches(
            driver,
            config["database"],
            UPSERT_MENTIONS,
            mention_rows(args.review_keyword_edges, args.limit),
            batch_size=args.batch_size,
            label="MENTIONS",
        )
        if args.skip_co_occurrence:
            summary["loaded_co_occurrences"] = 0
        else:
            summary["loaded_co_occurrences"] = run_write_batches(
                driver,
                config["database"],
                UPSERT_CO_OCCURRENCES,
                co_occurrence_rows(args.co_occurrence_edges, args.limit),
                batch_size=args.batch_size,
                label="CO_OCCURS_WITH",
            )

        summary["verification"] = verify(driver, config["database"])
        summary["elapsed_seconds"] = round(time.time() - started_at, 2)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_output.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
