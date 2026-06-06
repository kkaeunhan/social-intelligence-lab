from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT_DIR / "task2.2_network"
DATA_DIR = TASK_DIR / "data"

DEFAULT_INPUT = DATA_DIR / "context_candidate_clusters.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "context_labels.jsonl"
DEFAULT_PRODUCT_FEATURE_OUTPUT = DATA_DIR / "context_labels_product_feature.jsonl"
DEFAULT_PURCHASE_DELIVERY_OUTPUT = DATA_DIR / "context_labels_purchase_delivery_experience.jsonl"
DEFAULT_OTHER_OUTPUT = DATA_DIR / "context_labels_other.jsonl"
DEFAULT_SUMMARY = DATA_DIR / "context_label_summary.json"

ISSUE_TYPES = {
    "product_feature",
    "purchase_delivery_experience",
    "price_promotion",
    "cs_aftercare",
    "mixed_or_other",
}
ACTION_TYPES = {
    "marketing",
    "product_improvement",
    "cs",
    "logistics",
    "detail_page",
    "pricing_promotion",
    "monitoring",
}
PRIORITIES = {"high", "medium", "low"}
CONFIDENCES = {"high", "medium", "low"}

LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "cluster_id": {"type": "string"},
        "product_name": {"type": ["string", "null"]},
        "sentiment_id": {"type": ["string", "null"], "enum": ["positive", "neutral", "negative", None]},
        "issue_type": {
            "type": "string",
            "enum": [
                "product_feature",
                "purchase_delivery_experience",
                "price_promotion",
                "cs_aftercare",
                "mixed_or_other",
            ],
        },
        "context_label": {"type": "string"},
        "context_summary": {"type": "string"},
        "customer_perception": {"type": "string"},
        "evidence_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 12,
        },
        "evidence_pair_keys": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
        "representative_review_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "business_actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": [
                            "marketing",
                            "product_improvement",
                            "cs",
                            "logistics",
                            "detail_page",
                            "pricing_promotion",
                            "monitoring",
                        ],
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "rationale": {"type": "string"},
                },
                "required": ["action_type", "title", "description", "priority", "rationale"],
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "confidence_reason": {"type": "string"},
    },
    "required": [
        "cluster_id",
        "product_name",
        "sentiment_id",
        "issue_type",
        "context_label",
        "context_summary",
        "customer_perception",
        "evidence_keywords",
        "evidence_pair_keys",
        "representative_review_ids",
        "business_actions",
        "confidence",
        "confidence_reason",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM context labeling for Step 6 context candidate clusters."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--product-feature-output", type=Path, default=DEFAULT_PRODUCT_FEATURE_OUTPUT)
    parser.add_argument("--purchase-delivery-output", type=Path, default=DEFAULT_PURCHASE_DELIVERY_OUTPUT)
    parser.add_argument("--other-output", type=Path, default=DEFAULT_OTHER_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            if limit is not None and index > limit:
                break
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def system_prompt() -> str:
    return (
        "당신은 한국어 이커머스 리뷰를 분석하는 BI analyst입니다. "
        "PMI/NPMI 기반 keyword-pair 클러스터와 리뷰 근거만 사용해 고객 인식 맥락을 라벨링합니다. "
        "과장하지 말고, 표본이 작거나 증거가 혼합되어 있으면 confidence를 낮추세요. "
        "반드시 고정 JSON 스키마로만 답하세요."
    )


def user_prompt(cluster: dict[str, Any]) -> str:
    issue_rules = {
        "product_feature": "카메라, 배터리, 성능, 발열, 무게, 디자인, 화면, 저장공간 등 제품 자체 속성",
        "purchase_delivery_experience": "배송, 포장, 박스, 뽁뽁이, 파손 우려, 사전 구매 과정 등 구매/배송 경험",
        "price_promotion": "가격, 가성비, 할인, 카드 혜택, 사전예약 혜택, 라이브 방송 조건",
        "cs_aftercare": "교환, 환불, 반품, 서비스센터, 고객센터, 불량 대응",
        "mixed_or_other": "위 분류가 혼합되어 한쪽으로 명확히 분류하기 어려운 경우",
    }
    payload = {
        "classification_rules": issue_rules,
        "required_style": {
            "context_label": "한국어 명사구, 20자 내외",
            "context_summary": "근거 기반 1문장",
            "customer_perception": "고객이 제품/구매 경험을 어떻게 인식하는지 1문장",
            "business_actions": "마케팅, 제품개선, CS, 물류/포장, 상세페이지 중 실행 가능한 액션",
        },
        "cluster": cluster,
    }
    return json.dumps(payload, ensure_ascii=False)


def call_llm(client: OpenAI, model: str, cluster: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_completion_tokens=1200,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "context_label_result",
                "strict": True,
                "schema": LLM_OUTPUT_SCHEMA,
            },
        },
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(cluster)},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Empty LLM response for {cluster['cluster_id']}")
    return json.loads(content)


def normalize_list(values: list[Any], limit: int) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized


def validate_result(result: dict[str, Any], cluster: dict[str, Any]) -> dict[str, Any]:
    result["cluster_id"] = cluster["cluster_id"]
    result["product_name"] = cluster.get("product_name")
    result["sentiment_id"] = cluster.get("sentiment_id")
    if result["issue_type"] not in ISSUE_TYPES:
        result["issue_type"] = "mixed_or_other"
    if result["confidence"] not in CONFIDENCES:
        result["confidence"] = "low"

    pair_keys = [
        f"{pair['keyword_1']}||{pair['keyword_2']}"
        for pair in cluster.get("top_pairs") or []
        if pair.get("keyword_1") and pair.get("keyword_2")
    ]
    review_ids = [
        str(review["review_id"])
        for review in cluster.get("evidence_reviews") or []
        if review.get("review_id")
    ]
    evidence_keywords = []
    for pair in cluster.get("top_pairs") or []:
        evidence_keywords.extend([pair.get("keyword_1"), pair.get("keyword_2")])

    result["evidence_pair_keys"] = normalize_list(
        list(result.get("evidence_pair_keys") or []) + pair_keys,
        8,
    )
    result["representative_review_ids"] = normalize_list(
        list(result.get("representative_review_ids") or []) + review_ids,
        5,
    )
    result["evidence_keywords"] = normalize_list(
        list(result.get("evidence_keywords") or []) + evidence_keywords,
        12,
    )

    actions = []
    for index, action in enumerate(result.get("business_actions") or [], start=1):
        action_type = action.get("action_type")
        priority = action.get("priority")
        actions.append(
            {
                "action_id": stable_id(
                    "ba",
                    f"{cluster['cluster_id']}|{index}|{action_type}|{action.get('title')}",
                ),
                "action_type": action_type if action_type in ACTION_TYPES else "monitoring",
                "title": str(action.get("title") or "").strip(),
                "description": str(action.get("description") or "").strip(),
                "priority": priority if priority in PRIORITIES else "medium",
                "rationale": str(action.get("rationale") or "").strip(),
            }
        )
    result["business_actions"] = actions[:3]
    if not result["business_actions"]:
        result["business_actions"] = [
            {
                "action_id": stable_id("ba", f"{cluster['cluster_id']}|monitoring"),
                "action_type": "monitoring",
                "title": "추가 모니터링",
                "description": "근거 리뷰를 추가 확인해 반복 패턴 여부를 점검한다.",
                "priority": "low",
                "rationale": "실행 액션을 단정하기에는 증거가 제한적이다.",
            }
        ]
    return result


def enrich_result(result: dict[str, Any], cluster: dict[str, Any], model: str) -> dict[str, Any]:
    context_id = stable_id("ctx", cluster["cluster_id"])
    return {
        "context_id": context_id,
        "cluster_id": cluster["cluster_id"],
        "product_name": cluster.get("product_name"),
        "sentiment_id": cluster.get("sentiment_id"),
        "theme_hint": cluster.get("theme_hint"),
        "issue_type": result["issue_type"],
        "context_label": result["context_label"],
        "context_summary": result["context_summary"],
        "customer_perception": result["customer_perception"],
        "evidence_keywords": result["evidence_keywords"],
        "evidence_pair_keys": result["evidence_pair_keys"],
        "representative_review_ids": result["representative_review_ids"],
        "business_actions": result["business_actions"],
        "confidence": result["confidence"],
        "confidence_reason": result["confidence_reason"],
        "cluster_metrics": {
            "candidate_count": cluster.get("candidate_count"),
            "total_pair_count": cluster.get("total_pair_count"),
            "max_pair_count": cluster.get("max_pair_count"),
            "avg_npmi": cluster.get("avg_npmi"),
            "max_npmi": cluster.get("max_npmi"),
            "max_significance_score": cluster.get("max_significance_score"),
        },
        "top_pairs": cluster.get("top_pairs") or [],
        "evidence_reviews": cluster.get("evidence_reviews") or [],
        "llm_model": model,
        "created_at": int(time.time()),
    }


def split_outputs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    product_feature = [row for row in rows if row["issue_type"] == "product_feature"]
    purchase_delivery = [
        row for row in rows if row["issue_type"] == "purchase_delivery_experience"
    ]
    other = [
        row
        for row in rows
        if row["issue_type"] not in {"product_feature", "purchase_delivery_experience"}
    ]
    return product_feature, purchase_delivery, other


def summarize(rows: list[dict[str, Any]], model: str, elapsed_seconds: float) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in rows:
            key = str(row.get(field) or "null")
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    return {
        "model": model,
        "context_label_count": len(rows),
        "issue_type_counts": counts("issue_type"),
        "product_counts": counts("product_name"),
        "sentiment_counts": counts("sentiment_id"),
        "confidence_counts": counts("confidence"),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    model = args.model or os.getenv("OPENAI_METADATA_MODEL") or "gpt-4.1-mini"
    client = OpenAI(api_key=api_key)

    started_at = time.time()
    rows: list[dict[str, Any]] = []
    clusters = list(iter_jsonl(args.input, args.limit))
    for index, cluster in enumerate(clusters, start=1):
        raw_result = call_llm(client, model, cluster)
        result = validate_result(raw_result, cluster)
        rows.append(enrich_result(result, cluster, model))
        print(f"context label: {index}/{len(clusters)} {cluster['cluster_id']}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    product_feature, purchase_delivery, other = split_outputs(rows)
    write_jsonl(args.output, rows)
    write_jsonl(args.product_feature_output, product_feature)
    write_jsonl(args.purchase_delivery_output, purchase_delivery)
    write_jsonl(args.other_output, other)

    summary = summarize(rows, model, time.time() - started_at)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
