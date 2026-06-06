from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT_DIR / "task2.2_network"
DATA_DIR = TASK_DIR / "data"

DEFAULT_CONTEXT_LABELS = DATA_DIR / "context_labels.jsonl"
DEFAULT_SCHEMA = TASK_DIR / "cypher" / "schema.cypher"
DEFAULT_SUMMARY = DATA_DIR / "context_neo4j_load_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Step 6 context labels into Neo4j.")
    parser.add_argument("--context-labels", type=Path, default=DEFAULT_CONTEXT_LABELS)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--skip-schema", action="store_true")
    parser.add_argument(
        "--reset-context",
        action="store_true",
        help="Delete ContextLabel/BusinessAction nodes and their relationships before loading.",
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


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
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


def split_cypher_statements(text: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current).rstrip(";").strip())
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


def keyword_id(normalized: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z가-힣\s_]+", "", normalized.strip().lower())
    safe = re.sub(r"\s+", "_", safe)
    return f"kw:{safe}"


def pair_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in context.get("top_pairs") or []:
        keyword_1 = pair.get("keyword_1")
        keyword_2 = pair.get("keyword_2")
        if not keyword_1 or not keyword_2:
            continue
        rows.append(
            {
                "keyword_1": keyword_1,
                "keyword_2": keyword_2,
                "keyword_1_id": keyword_id(keyword_1),
                "keyword_2_id": keyword_id(keyword_2),
                "keyword_pair_key": f"{keyword_1}||{keyword_2}",
                "count": pair.get("count"),
                "npmi": pair.get("npmi"),
                "pmi": pair.get("pmi"),
                "significance_score": pair.get("significance_score"),
                "evidence_strength": pair.get("evidence_strength"),
            }
        )
    return rows


def context_rows(path: Path) -> Iterable[dict[str, Any]]:
    for context in iter_jsonl(path):
        cluster_metrics = context.get("cluster_metrics") or {}
        evidence_reviews = context.get("evidence_reviews") or []
        yield {
            "context_id": context["context_id"],
            "cluster_id": context["cluster_id"],
            "product_name": context.get("product_name"),
            "sentiment_id": context.get("sentiment_id"),
            "theme_hint": context.get("theme_hint"),
            "issue_type": context.get("issue_type"),
            "label": context.get("context_label"),
            "summary": context.get("context_summary"),
            "customer_perception": context.get("customer_perception"),
            "confidence": context.get("confidence"),
            "confidence_reason": context.get("confidence_reason"),
            "evidence_keywords": context.get("evidence_keywords") or [],
            "evidence_pair_keys": context.get("evidence_pair_keys") or [],
            "representative_review_ids": context.get("representative_review_ids") or [],
            "candidate_count": cluster_metrics.get("candidate_count"),
            "total_pair_count": cluster_metrics.get("total_pair_count"),
            "max_pair_count": cluster_metrics.get("max_pair_count"),
            "avg_npmi": cluster_metrics.get("avg_npmi"),
            "max_npmi": cluster_metrics.get("max_npmi"),
            "max_significance_score": cluster_metrics.get("max_significance_score"),
            "top_pairs_json": json.dumps(context.get("top_pairs") or [], ensure_ascii=False),
            "evidence_reviews_json": json.dumps(evidence_reviews, ensure_ascii=False),
            "llm_model": context.get("llm_model"),
            "created_at": context.get("created_at"),
            "top_pairs": pair_rows(context),
            "business_actions": context.get("business_actions") or [],
        }


UPSERT_CONTEXTS = """
UNWIND $rows AS row
MERGE (c:ContextLabel {context_id: row.context_id})
SET c.cluster_id = row.cluster_id,
    c.product_name = row.product_name,
    c.sentiment_id = row.sentiment_id,
    c.theme_hint = row.theme_hint,
    c.issue_type = row.issue_type,
    c.label = row.label,
    c.summary = row.summary,
    c.customer_perception = row.customer_perception,
    c.confidence = row.confidence,
    c.confidence_reason = row.confidence_reason,
    c.evidence_keywords = row.evidence_keywords,
    c.evidence_pair_keys = row.evidence_pair_keys,
    c.representative_review_ids = row.representative_review_ids,
    c.candidate_count = row.candidate_count,
    c.total_pair_count = row.total_pair_count,
    c.max_pair_count = row.max_pair_count,
    c.avg_npmi = row.avg_npmi,
    c.max_npmi = row.max_npmi,
    c.max_significance_score = row.max_significance_score,
    c.top_pairs_json = row.top_pairs_json,
    c.evidence_reviews_json = row.evidence_reviews_json,
    c.model = row.llm_model,
    c.created_at = row.created_at
WITH row, c
MATCH (p:Product {product_name: row.product_name})
MERGE (p)-[hc:HAS_CONTEXT]->(c)
SET hc.issue_type = row.issue_type,
    hc.sentiment_id = row.sentiment_id,
    hc.confidence = row.confidence
WITH row, c
MATCH (s:Sentiment {sentiment_id: row.sentiment_id})
MERGE (c)-[is:IN_SENTIMENT]->(s)
SET is.rule_version = "rating_v1"
WITH row, c
UNWIND row.representative_review_ids AS review_id
MATCH (rv:Review {review_id: review_id})
MERGE (c)-[sb:SUPPORTED_BY]->(rv)
SET sb.source = "llm_context_label"
"""

UPSERT_CONTEXT_KEYWORDS = """
UNWIND $rows AS row
MATCH (c:ContextLabel {context_id: row.context_id})
UNWIND row.top_pairs AS pair
MATCH (k1:Keyword {keyword_id: pair.keyword_1_id})
MATCH (k2:Keyword {keyword_id: pair.keyword_2_id})
MERGE (c)-[r1:EXPLAINS_PAIR {keyword_pair_key: pair.keyword_pair_key, role: "source"}]->(k1)
SET r1.keyword_1 = pair.keyword_1,
    r1.keyword_2 = pair.keyword_2,
    r1.count = pair.count,
    r1.npmi = pair.npmi,
    r1.pmi = pair.pmi,
    r1.significance_score = pair.significance_score,
    r1.evidence_strength = pair.evidence_strength
MERGE (c)-[r2:EXPLAINS_PAIR {keyword_pair_key: pair.keyword_pair_key, role: "target"}]->(k2)
SET r2.keyword_1 = pair.keyword_1,
    r2.keyword_2 = pair.keyword_2,
    r2.count = pair.count,
    r2.npmi = pair.npmi,
    r2.pmi = pair.pmi,
    r2.significance_score = pair.significance_score,
    r2.evidence_strength = pair.evidence_strength
"""

UPSERT_ACTIONS = """
UNWIND $rows AS row
MATCH (c:ContextLabel {context_id: row.context_id})
UNWIND row.business_actions AS action
MERGE (a:BusinessAction {action_id: action.action_id})
SET a.action_type = action.action_type,
    a.title = action.title,
    a.description = action.description,
    a.priority = action.priority,
    a.context_id = row.context_id,
    a.product_name = row.product_name,
    a.sentiment_id = row.sentiment_id,
    a.issue_type = row.issue_type,
    a.created_at = row.created_at
MERGE (c)-[m:MAPS_TO]->(a)
SET m.rationale = action.rationale,
    m.priority = action.priority,
    m.action_type = action.action_type
"""

RESET_CONTEXT = """
MATCH (n)
WHERE n:ContextLabel OR n:BusinessAction
DETACH DELETE n
"""

VERIFY_CONTEXT = """
CALL () {
  MATCH (c:ContextLabel) RETURN "ContextLabel" AS name, count(c) AS count
  UNION ALL
  MATCH (a:BusinessAction) RETURN "BusinessAction" AS name, count(a) AS count
  UNION ALL
  MATCH ()-[r:HAS_CONTEXT]->() RETURN "HAS_CONTEXT" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:IN_SENTIMENT]->() RETURN "IN_SENTIMENT" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:SUPPORTED_BY]->() RETURN "SUPPORTED_BY" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:EXPLAINS_PAIR]->() RETURN "EXPLAINS_PAIR" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:MAPS_TO]->() RETURN "MAPS_TO" AS name, count(r) AS count
}
RETURN name, count
ORDER BY name
"""

VERIFY_ISSUE_TYPES = """
MATCH (c:ContextLabel)
RETURN c.issue_type AS issue_type, count(c) AS count
ORDER BY count DESC, issue_type
"""


def reset_context(driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.execute_write(lambda tx: tx.run(RESET_CONTEXT).consume())


def verify(driver, database: str) -> dict[str, Any]:
    with driver.session(database=database) as session:
        counts = {record["name"]: record["count"] for record in session.run(VERIFY_CONTEXT)}
        issue_types = [dict(record) for record in session.run(VERIFY_ISSUE_TYPES)]
    return {
        "counts": counts,
        "issue_types": issue_types,
    }


def main() -> None:
    args = parse_args()
    started_at = time.time()
    config = load_neo4j_config()
    rows = list(context_rows(args.context_labels))
    summary: dict[str, Any] = {
        "database": config["database"],
        "input_file": str(args.context_labels),
        "batch_size": args.batch_size,
        "input_context_label_count": len(rows),
    }

    driver = GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"]))
    try:
        driver.verify_connectivity()
        if not args.skip_schema:
            summary["schema_statements_executed"] = run_schema(
                driver, config["database"], args.schema
            )
        else:
            summary["schema_statements_executed"] = 0

        if args.reset_context:
            reset_context(driver, config["database"])
        summary["reset_context"] = args.reset_context

        summary["loaded_context_labels"] = run_write_batches(
            driver,
            config["database"],
            UPSERT_CONTEXTS,
            rows,
            batch_size=args.batch_size,
            label="ContextLabel",
        )
        summary["loaded_context_keyword_links"] = run_write_batches(
            driver,
            config["database"],
            UPSERT_CONTEXT_KEYWORDS,
            rows,
            batch_size=args.batch_size,
            label="EXPLAINS_PAIR",
        )
        summary["loaded_business_actions"] = run_write_batches(
            driver,
            config["database"],
            UPSERT_ACTIONS,
            rows,
            batch_size=args.batch_size,
            label="BusinessAction/MAPS_TO",
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
