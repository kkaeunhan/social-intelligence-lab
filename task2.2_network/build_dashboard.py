from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT_DIR / "task2.2_network"
DATA_DIR = TASK_DIR / "data"
DEFAULT_OUTPUT = TASK_DIR / "dashboard" / "network_dashboard.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained dashboard from Task 2.2 network outputs."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-edges", type=int, default=28)
    parser.add_argument("--top-contexts", type=int, default=10)
    parser.add_argument("--top-actions", type=int, default=12)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def top_global_edges(data_dir: Path, limit: int) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    with (data_dir / "co_occurrence_edges.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("scope") != "global":
                continue
            if row.get("co_count", 0) < 5:
                continue
            edges.append(
                {
                    "source": row["keyword_1"],
                    "target": row["keyword_2"],
                    "count": row["co_count"],
                    "npmi": row["npmi"],
                    "pmi": row["pmi"],
                    "lift": row["lift"],
                }
            )
            if len(edges) >= limit:
                break
    return edges


def load_contexts(
    data_dir: Path, context_limit: int, action_limit: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = read_jsonl(data_dir / "context_labels.jsonl")
    labels.sort(
        key=lambda row: (
            row.get("cluster_metrics", {}).get("max_significance_score", 0),
            row.get("cluster_metrics", {}).get("total_pair_count", 0),
        ),
        reverse=True,
    )

    action_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    action_rows: list[dict[str, Any]] = []
    for row in labels:
        matrix[row["product_name"]][row["issue_type"]] += 1
        for action in row.get("business_actions", []):
            action_counts[action.get("action_type", "unknown")] += 1
            priority_counts[action.get("priority", "unknown")] += 1
            metrics = row.get("cluster_metrics", {})
            top_pair = row.get("top_pairs", [{}])[0] if row.get("top_pairs") else {}
            action_rows.append(
                {
                    "product_name": row["product_name"],
                    "sentiment_id": row["sentiment_id"],
                    "issue_type": row["issue_type"],
                    "theme_hint": row["theme_hint"],
                    "context_label": row["context_label"],
                    "customer_perception": row.get("customer_perception", ""),
                    "action_type": action.get("action_type", "unknown"),
                    "priority": action.get("priority", "unknown"),
                    "title": action.get("title", ""),
                    "description": action.get("description", ""),
                    "rationale": action.get("rationale", ""),
                    "total_pair_count": metrics.get("total_pair_count", 0),
                    "max_npmi": metrics.get("max_npmi", 0),
                    "significance": metrics.get("max_significance_score", 0),
                    "pair": (
                        f"{top_pair.get('keyword_1', '')} ↔ {top_pair.get('keyword_2', '')}"
                        if top_pair
                        else ""
                    ),
                }
            )

    contexts = []
    for row in labels[:context_limit]:
        actions = row.get("business_actions", [])
        metrics = row.get("cluster_metrics", {})
        contexts.append(
            {
                "product_name": row["product_name"],
                "sentiment_id": row["sentiment_id"],
                "theme_hint": row["theme_hint"],
                "issue_type": row["issue_type"],
                "context_label": row["context_label"],
                "summary": row["context_summary"],
                "confidence": row["confidence"],
                "total_pair_count": metrics.get("total_pair_count", 0),
                "max_npmi": metrics.get("max_npmi", 0),
                "max_significance_score": metrics.get("max_significance_score", 0),
                "top_pairs": [
                    {
                        "keyword_1": pair["keyword_1"],
                        "keyword_2": pair["keyword_2"],
                        "count": pair["count"],
                        "npmi": pair["npmi"],
                    }
                    for pair in row.get("top_pairs", [])[:3]
                ],
                "actions": [
                    {
                        "action_type": action.get("action_type"),
                        "title": action.get("title"),
                        "priority": action.get("priority"),
                    }
                    for action in actions
                ],
                "evidence_excerpt": (
                    row.get("evidence_reviews", [{}])[0].get("excerpt", "")
                    if row.get("evidence_reviews")
                    else ""
                ),
            }
        )

    issue_types = sorted({issue for counts in matrix.values() for issue in counts})
    matrix_rows = [
        {
            "product_name": product,
            **{issue: matrix[product][issue] for issue in issue_types},
        }
        for product in sorted(matrix)
    ]

    priority_rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    action_rows.sort(
        key=lambda row: (
            priority_rank.get(row["priority"], 3),
            -row["significance"],
            -row["total_pair_count"],
        )
    )

    return contexts, {
        "action_counts": dict(action_counts),
        "priority_counts": dict(priority_counts),
        "issue_types": issue_types,
        "product_issue_matrix": matrix_rows,
        "actions": action_rows[:action_limit],
        "labels": labels,
    }


def derive_insights(
    candidate: dict[str, Any],
    label: dict[str, Any],
    context_rollup: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = context_rollup["labels"]

    def matching(**conditions: str) -> list[dict[str, Any]]:
        rows = labels
        for key, value in conditions.items():
            rows = [row for row in rows if row.get(key) == value]
        return rows

    def product_list(rows: list[dict[str, Any]], max_count: int = 4) -> str:
        products = sorted({row["product_name"] for row in rows})
        if len(products) <= max_count:
            return ", ".join(products)
        return ", ".join(products[:max_count]) + f" 외 {len(products) - max_count}개"

    packaging = [
        row
        for row in labels
        if row.get("issue_type") == "purchase_delivery_experience"
        and row.get("theme_hint") == "배송/포장 리스크"
    ]
    packaging_negative = [row for row in packaging if row.get("sentiment_id") == "negative"]
    promo = matching(issue_type="price_promotion")
    feature = matching(issue_type="product_feature")
    cs = matching(issue_type="cs_aftercare")

    action_counts = context_rollup["action_counts"]
    top_action, top_action_count = max(action_counts.items(), key=lambda item: item[1])

    return [
        {
            "title": "포장/배송은 제품 기능과 분리해서 관리해야 하는 리스크",
            "metric": f"{len(packaging)}개 context",
            "evidence": (
                f"배송/포장 리스크 후보 {candidate['candidate_counts_by_theme'].get('배송/포장 리스크', 0)}개, "
                f"부정 context {len(packaging_negative)}개. 관련 제품: {product_list(packaging)}"
            ),
            "value": "상품 자체 만족도와 별개로 구매 경험에서 불만이 생기는 구간을 분리해 CS, 포장 기준, 상세페이지 안내로 연결할 수 있다.",
        },
        {
            "title": "사전예약/혜택은 긍정 구매 맥락을 만드는 핵심 메시지",
            "metric": f"{len(promo)}개 context",
            "evidence": (
                "Global pair '사전 ↔ 예약' NPMI 0.866, count 250. "
                f"관련 제품: {product_list(promo)}"
            ),
            "value": "가격 자체보다 '혜택을 잘 받았다'는 인식을 상세페이지와 라이브/예약 캠페인 메시지로 재활용할 수 있다.",
        },
        {
            "title": "제품 기능 이슈는 카메라, 성능, 무게, 디자인으로 구체화됨",
            "metric": f"{len(feature)}개 context",
            "evidence": (
                f"전체 issue type 중 product_feature가 {label['issue_type_counts'].get('product_feature', 0)}개로 최다. "
                f"관련 제품: {product_list(feature)}"
            ),
            "value": "단순 별점 요약이 아니라 제품별 강점/약점을 키워드 pair와 리뷰 근거로 설명해 마케팅 문구와 개선 포인트를 나눌 수 있다.",
        },
        {
            "title": "드문 CS 이슈도 우선순위 높은 조기 경보로 볼 수 있음",
            "metric": f"{len(cs)}개 context",
            "evidence": (
                f"cs_aftercare context {len(cs)}개, high priority action "
                f"{sum(1 for row in cs for action in row.get('business_actions', []) if action.get('priority') == 'high')}개."
            ),
            "value": "빈도는 작아도 불량/교환/환불은 브랜드 신뢰에 직접 영향을 주므로 모니터링 항목으로 분리하는 근거가 된다.",
        },
        {
            "title": "LLM 결과가 바로 실행 가능한 action taxonomy로 변환됨",
            "metric": f"{top_action_count}개 {top_action}",
            "evidence": (
                f"Business action 총 {sum(action_counts.values())}개. "
                f"가장 많은 action type은 {top_action}."
            ),
            "value": "리뷰 분석 결과를 '마케팅/상세페이지/CS/제품개선' 단위로 배분해 담당 조직별 후속 작업으로 넘길 수 있다.",
        },
    ]


def derive_product_playbook(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        grouped[row["product_name"]].append(row)

    playbook = []
    confidence_rank = {"high": 2, "medium": 1, "low": 0}
    for product, rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda row: (
                confidence_rank.get(row.get("confidence", "medium"), 1),
                row.get("cluster_metrics", {}).get("max_significance_score", 0),
            ),
            reverse=True,
        )
        positive = [
            row
            for row in rows
            if row.get("sentiment_id") == "positive"
            and row.get("issue_type") in {"product_feature", "price_promotion"}
        ]
        risks = [
            row
            for row in rows
            if row.get("sentiment_id") in {"negative", "neutral"}
            or row.get("issue_type") in {"purchase_delivery_experience", "cs_aftercare"}
        ]
        actions = [
            action
            for row in rows
            for action in row.get("business_actions", [])
            if action.get("priority") == "high"
        ]
        top_positive = positive[0] if positive else rows[0]
        top_risk = risks[0] if risks else rows[-1]
        playbook.append(
            {
                "product_name": product,
                "context_count": len(rows),
                "high_priority_actions": len(actions),
                "top_opportunity": top_positive.get("context_label", ""),
                "opportunity_value": top_positive.get("customer_perception", ""),
                "top_risk": top_risk.get("context_label", ""),
                "risk_value": top_risk.get("customer_perception", ""),
                "recommended_action": actions[0].get("title", "") if actions else "",
            }
        )
    return playbook


def derive_value_findings(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    action_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issue_by_action: dict[str, Counter[str]] = defaultdict(Counter)
    for row in labels:
        for action in row.get("business_actions", []):
            action_by_type[action.get("action_type", "unknown")].append(row)
            issue_by_action[action.get("action_type", "unknown")][row["issue_type"]] += 1

    for action_type, rows in sorted(action_by_type.items(), key=lambda item: len(item[1]), reverse=True):
        top_issue, top_issue_count = issue_by_action[action_type].most_common(1)[0]
        products = sorted({row["product_name"] for row in rows})
        examples = sorted(
            rows,
            key=lambda row: row.get("cluster_metrics", {}).get("max_significance_score", 0),
            reverse=True,
        )[:2]
        findings.append(
            {
                "action_type": action_type,
                "count": len(rows),
                "dominant_issue": top_issue,
                "dominant_issue_count": top_issue_count,
                "products": ", ".join(products[:4]) + (" 외" if len(products) > 4 else ""),
                "examples": [row["context_label"] for row in examples],
                "meaning": action_meaning(action_type, top_issue),
            }
        )
    return findings


def derive_stakeholder_insights(labels: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    action_counts: Counter[str] = Counter()
    high_action_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for row in labels:
        for action in row.get("business_actions", []):
            action_counts[action.get("action_type", "unknown")] += 1
            if action.get("priority") == "high":
                high_action_rows.append((row, action))

    marketing_count = action_counts.get("marketing", 0)
    detail_count = action_counts.get("detail_page", 0)
    cs_count = action_counts.get("cs", 0)
    improvement_count = action_counts.get("product_improvement", 0)
    delivery_contexts = [
        row
        for row in labels
        if row["issue_type"] == "purchase_delivery_experience"
        or row["theme_hint"] == "배송/포장 리스크"
    ]
    product_feature_contexts = [row for row in labels if row["issue_type"] == "product_feature"]
    price_contexts = [row for row in labels if row["issue_type"] == "price_promotion"]
    consumer_feature_contexts = [
        row
        for row in product_feature_contexts
        if row["theme_hint"] in {"카메라/화질", "성능/속도", "무게/휴대성/그립", "디자인/색상/마감", "배터리/충전"}
    ]
    camera_contexts = [
        row for row in consumer_feature_contexts if row["theme_hint"] == "카메라/화질"
    ]
    best_feature = best_context_detail(product_feature_contexts)
    best_delivery = best_context_detail(delivery_contexts)
    best_price = best_context_detail(price_contexts)
    best_consumer_feature = best_context_detail(camera_contexts or consumer_feature_contexts)

    enterprise = [
        {
            "title": "리뷰 속 강점을 광고 메시지로 전환",
            "metric": f"{marketing_count} marketing actions",
            "signal": f"제품 기능 context {len(product_feature_contexts)}개",
            "value": "반복 언급된 제품 강점을 캠페인 문구로 바로 활용할 수 있음",
            "example": best_feature["example"],
            "detail": best_feature["detail"],
        },
        {
            "title": "포장/배송 불만을 CS 개선 과제로 분리",
            "metric": f"{detail_count + cs_count + improvement_count} CX actions",
            "signal": f"배송·구매 경험 context {len(delivery_contexts)}개",
            "value": "제품 성능과 별개인 구매 경험 리스크를 운영 개선 항목으로 관리",
            "example": best_delivery["example"],
            "detail": best_delivery["detail"],
        },
    ]

    consumer = [
        {
            "title": "내가 중요하게 보는 기준과 제품 강점 매칭",
            "metric": f"{len(labels)} context labels",
            "signal": "카메라·성능·무게·디자인 기준 비교",
            "value": "별점보다 구체적인 사용 맥락으로 제품 선택 기준을 확인",
            "example": best_consumer_feature["example"],
            "detail": best_consumer_feature["detail"],
        },
        {
            "title": "혜택 만족과 배송 리스크를 함께 확인",
            "metric": f"{len(price_contexts)} benefit contexts",
            "signal": f"배송·구매 경험 context {len(delivery_contexts)}개와 비교",
            "value": "사전예약·할인 혜택의 체감 가치와 구매 전 주의점을 동시에 판단",
            "example": best_price["example"],
            "detail": best_price["detail"],
        },
    ]
    return {"enterprise": enterprise, "consumer": consumer}


def best_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    row = max(
        rows,
        key=lambda item: item.get("cluster_metrics", {}).get("max_significance_score", 0),
    )
    return f"{row['product_name']} - {row['context_label']}"


def best_context_detail(rows: list[dict[str, Any]]) -> dict[str, str]:
    if not rows:
        return {"example": "", "detail": ""}
    row = max(
        rows,
        key=lambda item: item.get("cluster_metrics", {}).get("max_significance_score", 0),
    )
    pair = row.get("top_pairs", [{}])[0] if row.get("top_pairs") else {}
    metrics = row.get("cluster_metrics", {})
    pair_text = (
        f"{pair.get('keyword_1', '')} ↔ {pair.get('keyword_2', '')}"
        if pair
        else "대표 pair 없음"
    )
    detail = (
        f"{row.get('context_label', '')} / {pair_text} / "
        f"NPMI {metrics.get('max_npmi', 0):.2f}, pair count {metrics.get('total_pair_count', 0)}"
    )
    return {"example": f"{row['product_name']} - {row['theme_hint']}", "detail": detail}


def action_meaning(action_type: str, issue_type: str) -> str:
    meanings = {
        "marketing": "긍정 인식이 강한 기능/혜택을 광고와 상세페이지 메시지로 전환할 수 있다.",
        "detail_page": "구매 전 불안을 낮추는 정보 보강 포인트를 찾은 것이다.",
        "product_improvement": "반복되는 불만 맥락을 제품/운영 개선 backlog로 넘길 수 있다.",
        "cs": "리뷰 기반으로 CS 응대 스크립트와 보상 기준을 정교화할 수 있다.",
        "monitoring": "빈도는 낮지만 리스크가 큰 신호를 지속 관찰 대상으로 분리할 수 있다.",
        "logistics": "상품 경험과 분리된 배송/포장 운영 개선 포인트를 찾은 것이다.",
        "pricing_promotion": "가격보다 혜택 인식을 강화하는 프로모션 설계 근거가 된다.",
    }
    return meanings.get(action_type, f"{issue_type} 이슈를 실행 가능한 후속 작업으로 분류했다.")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir
    preprocess = read_json(data_dir / "preprocess_summary.json")
    keyword = read_json(data_dir / "keyword_extraction_summary.json")
    pmi = read_json(data_dir / "pmi_npmi_summary.json")
    candidate = read_json(data_dir / "context_candidate_summary.json")
    label = read_json(data_dir / "context_label_summary.json")
    context_neo4j = read_json(data_dir / "context_neo4j_load_summary.json")
    contexts, context_rollup = load_contexts(data_dir, args.top_contexts, args.top_actions)
    insights = derive_insights(candidate, label, context_rollup)
    playbook = derive_product_playbook(context_rollup["labels"])
    value_findings = derive_value_findings(context_rollup["labels"])
    stakeholder_insights = derive_stakeholder_insights(context_rollup["labels"])
    context_rollup = {key: value for key, value in context_rollup.items() if key != "labels"}

    return {
        "preprocess": preprocess,
        "keyword": keyword,
        "pmi": pmi,
        "candidate": candidate,
        "label": label,
        "context_neo4j": context_neo4j,
        "edges": top_global_edges(data_dir, args.top_edges),
        "contexts": contexts,
        "context_rollup": context_rollup,
        "insights": insights,
        "playbook": playbook,
        "value_findings": value_findings,
        "stakeholder_insights": stakeholder_insights,
    }


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Task 2.2 Network Analysis Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #18212f;
      --muted: #657184;
      --line: #dce3ec;
      --blue: #2563eb;
      --green: #15803d;
      --red: #dc2626;
      --amber: #b45309;
      --violet: #7c3aed;
      --cyan: #0891b2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, "Noto Sans KR", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      background: #101827;
      color: white;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    p {{ margin: 0; color: var(--muted); }}
    header p {{ color: #c8d3e2; max-width: 820px; font-size: 15px; }}
    main {{ padding: 22px 32px 42px; }}
    .grid {{ display: grid; gap: 14px; }}
    .metrics {{ grid-template-columns: repeat(6, minmax(130px, 1fr)); margin-bottom: 18px; }}
    .two {{ grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr); }}
    .three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    section, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ font-size: 27px; font-weight: 700; margin-top: 4px; }}
    .metric .note {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
    .insights {{ grid-template-columns: repeat(5, minmax(190px, 1fr)); margin-bottom: 14px; }}
    .insight-card {{
      border-left: 4px solid var(--blue);
      min-height: 190px;
    }}
    .insight-card:nth-child(2) {{ border-left-color: var(--green); }}
    .insight-card:nth-child(3) {{ border-left-color: var(--violet); }}
    .insight-card:nth-child(4) {{ border-left-color: var(--red); }}
    .insight-card:nth-child(5) {{ border-left-color: var(--amber); }}
    .insight-metric {{ font-size: 22px; font-weight: 700; margin: 4px 0 8px; }}
    .insight-title {{ font-weight: 700; margin-bottom: 6px; }}
    .insight-text {{ color: #334155; font-size: 13px; margin-top: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: 168px 1fr 48px; gap: 10px; align-items: center; margin: 9px 0; }}
    .bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }}
    .bar-track {{ background: #edf1f6; height: 12px; border-radius: 6px; overflow: hidden; }}
    .bar {{ height: 100%; background: var(--blue); border-radius: 6px; }}
    .bar-num {{ text-align: right; color: var(--muted); font-size: 12px; }}
    canvas {{ width: 100%; height: 430px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcff; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 7px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; background: #f5f7fa; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 3px 8px; font-size: 12px; background: #eef2ff; color: #3730a3; }}
    .positive {{ background: #dcfce7; color: #166534; }}
    .negative {{ background: #fee2e2; color: #991b1b; }}
    .neutral {{ background: #fef3c7; color: #92400e; }}
    .context-list {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; max-height: 620px; overflow: auto; padding-right: 2px; }}
    .context-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }}
    .context-meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
    .context-title {{ font-weight: 800; font-size: 15px; margin-bottom: 5px; }}
    .context-summary {{ color: #334155; font-size: 13px; margin-bottom: 8px; }}
    .evidence {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
    .action-board {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; max-height: 660px; overflow: auto; padding-right: 2px; }}
    .action-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }}
    .action-head {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 9px; }}
    .action-title {{ font-weight: 800; font-size: 15px; margin-bottom: 5px; }}
    .action-desc {{ color: #334155; font-size: 13px; line-height: 1.42; margin-bottom: 8px; }}
    .section-subtitle {{ color: var(--muted); font-size: 13px; margin: -6px 0 12px; }}
    .field-label {{ color: #475569; font-weight: 700; }}
    .playbook-cell {{ min-width: 180px; }}
    .small {{ color: var(--muted); font-size: 12px; }}
    .stakeholder-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .stakeholder-card {{
      min-height: 168px;
      padding: 18px;
      display: grid;
      gap: 8px;
    }}
    .stakeholder-card.enterprise {{ border-top: 4px solid var(--blue); }}
    .stakeholder-card.consumer {{ border-top: 4px solid var(--green); }}
    .stakeholder-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .stakeholder-label {{ color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; }}
    .stakeholder-title {{
      font-weight: 800;
      font-size: 20px;
      line-height: 1.25;
      max-width: 70%;
      overflow-wrap: anywhere;
    }}
    .stakeholder-metric {{
      color: var(--blue);
      font-size: 22px;
      font-weight: 800;
      line-height: 1.1;
      text-align: right;
      white-space: nowrap;
    }}
    .stakeholder-card.consumer .stakeholder-metric {{ color: var(--green); }}
    .stakeholder-signal {{
      display: inline-block;
      width: fit-content;
      max-width: 100%;
      color: #1f2937;
      background: #f1f5f9;
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 13px;
      font-weight: 700;
    }}
    .stakeholder-card p {{ color: #334155; font-size: 14px; font-weight: 600; }}
    .stakeholder-proof {{
      border-top: 1px solid var(--line);
      padding-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; color: var(--muted); font-size: 12px; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 4px; }}
    @media (max-width: 1050px) {{
      .metrics, .three, .insights, .stakeholder-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .two {{ grid-template-columns: 1fr; }}
      .action-board, .context-list {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 640px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .metrics, .three, .insights, .stakeholder-grid {{ grid-template-columns: 1fr; }}
      .action-board, .context-list {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 118px 1fr 42px; }}
      canvas {{ height: 360px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>스마트폰 리뷰 연관어 네트워크 대시보드</h1>
    <p>리뷰 속 연관어와 근거 문장을 연결해 제품 강점, 구매 리스크, 실행 액션을 빠르게 확인하는 분석 결과 화면</p>
  </header>
  <main>
    <section style="margin-bottom:14px;">
      <h2>전략적 시사점</h2>
      <div id="enterpriseInsights" class="grid stakeholder-grid"></div>
    </section>
    <section style="margin-bottom:14px;">
      <h2>구매 의사결정 신호</h2>
      <div id="consumerInsights" class="grid stakeholder-grid"></div>
    </section>
    <section>
      <h2>Context Label 분포</h2>
      <div id="issueBars"></div>
    </section>
    <div class="grid two" style="margin-top:14px;">
      <section>
        <h2>Global PMI/NPMI 키워드 네트워크</h2>
        <canvas id="network" width="1000" height="520"></canvas>
        <div class="legend">
          <span><i class="swatch" style="background:#2563eb"></i>높은 빈도/중심 키워드</span>
          <span><i class="swatch" style="background:#dc2626"></i>높은 NPMI 연결</span>
          <span><i class="swatch" style="background:#64748b"></i>낮은 NPMI 연결</span>
          <span>edge 두께 = 동시 등장 count</span>
        </div>
      </section>
      <section>
        <h2>상위 연관어 Pair</h2>
        <table id="edgeTable"></table>
      </section>
    </div>
    <section style="margin-top:14px;">
      <h2>Action Board</h2>
      <p class="section-subtitle">LLM이 생성한 BusinessAction 중 우선순위가 높은 실행 항목을 context와 근거 pair와 함께 정리했습니다.</p>
      <div id="actionBoard" class="action-board"></div>
    </section>
    <section style="margin-top:14px;">
      <h2>Evidence Drill-down</h2>
      <p class="section-subtitle">ContextLabel별 요약, 연관어 pair, 대표 리뷰를 확인하는 근거 확인 영역입니다.</p>
      <div id="contexts" class="context-list"></div>
    </section>
  </main>
  <script>
    const data = {data};
    const fmt = new Intl.NumberFormat("ko-KR");
    const colors = ["#2563eb", "#15803d", "#dc2626", "#b45309", "#7c3aed", "#0891b2"];

    function metric(label, value, note) {{
      return `<div class="card metric"><div class="label">${{label}}</div><div class="value">${{fmt.format(value)}}</div><div class="note">${{note}}</div></div>`;
    }}

    function bars(id, obj, color = "#2563eb") {{
      const entries = Object.entries(obj).sort((a, b) => b[1] - a[1]);
      const max = Math.max(...entries.map(([, v]) => v), 1);
      document.getElementById(id).innerHTML = entries.map(([k, v]) => `
        <div class="bar-row">
          <div class="bar-label" title="${{k}}">${{k}}</div>
          <div class="bar-track"><div class="bar" style="width:${{(v / max) * 100}}%; background:${{color}}"></div></div>
          <div class="bar-num">${{fmt.format(v)}}</div>
        </div>`).join("");
    }}

    function renderMetrics() {{
      const p = data.preprocess;
      const k = data.keyword;
      const n = data.pmi;
      const c = data.candidate;
      const l = data.label;
      document.getElementById("metrics").innerHTML = [
        metric("정제 리뷰", p.cleaned_records, `${{p.raw_records}}개 원본 중 분석 가능 리뷰`),
        metric("고유 키워드", k.keyword_count, `Review-Keyword edge ${{fmt.format(k.edge_count)}}개`),
        metric("Co-occurrence", n.co_occurrence_edge_count, "PMI/NPMI 계산 edge"),
        metric("Context 후보", c.candidate_count, `${{c.cluster_count}}개 LLM-ready cluster`),
        metric("Context Label", l.context_label_count, `${{l.confidence_counts.high}} high / ${{l.confidence_counts.medium}} medium`),
        metric("Business Action", Object.values(data.context_rollup.action_counts).reduce((a,b) => a + b, 0), `Neo4j 적재 count ${{data.context_neo4j.verification.counts.BusinessAction}}`),
      ].join("");
    }}

    function renderInsights() {{
      document.getElementById("insights").innerHTML = data.insights.map(item => `
        <div class="card insight-card">
          <div class="insight-title">${{item.title}}</div>
          <div class="insight-metric">${{item.metric}}</div>
          <p>${{item.evidence}}</p>
          <div class="insight-text">${{item.value}}</div>
        </div>
      `).join("");
    }}

    function renderStakeholderInsights() {{
      const render = (items, cls, label) => items.map(item => `
        <div class="card stakeholder-card ${{cls}}">
          <div class="stakeholder-label">${{label}}</div>
          <div class="stakeholder-top">
            <div class="stakeholder-title">${{item.title}}</div>
            <div class="stakeholder-metric">${{item.metric}}</div>
          </div>
          <div class="stakeholder-signal">${{item.signal}}</div>
          <p>${{item.value}}</p>
          <div class="stakeholder-proof">${{item.example}}<br>${{item.detail}}</div>
        </div>
      `).join("");
      document.getElementById("enterpriseInsights").innerHTML = render(data.stakeholder_insights.enterprise, "enterprise", "strategy");
      document.getElementById("consumerInsights").innerHTML = render(data.stakeholder_insights.consumer, "consumer", "decision");
    }}

    function pill(text) {{
      const cls = text === "positive" ? "positive" : text === "negative" ? "negative" : text === "neutral" ? "neutral" : "";
      return `<span class="pill ${{cls}}">${{text}}</span>`;
    }}

    function renderTables() {{
      document.getElementById("edgeTable").innerHTML = `
        <thead><tr><th>keyword pair</th><th>count</th><th>NPMI</th><th>PMI</th></tr></thead>
        <tbody>${{data.edges.slice(0, 14).map(e => `
          <tr><td>${{e.source}} ↔ ${{e.target}}</td><td>${{fmt.format(e.count)}}</td><td>${{e.npmi.toFixed(3)}}</td><td>${{e.pmi.toFixed(2)}}</td></tr>
        `).join("")}}</tbody>`;

    }}

    function renderActionBoard() {{
      document.getElementById("actionBoard").innerHTML = data.context_rollup.actions.map(action => `
        <div class="action-card">
          <div class="action-head">
            ${{pill(action.product_name)}} ${{pill(action.sentiment_id)}} ${{pill(action.issue_type)}} ${{pill(action.action_type)}} ${{pill(action.priority)}}
          </div>
          <div class="action-title">${{action.title}}</div>
          <div class="action-desc">${{action.description}}</div>
          <div class="evidence">
            <span class="field-label">Context</span> ${{action.context_label}}<br>
            <span class="field-label">Evidence</span> ${{action.pair}} / NPMI ${{Number(action.max_npmi).toFixed(2)}} / count ${{fmt.format(action.total_pair_count)}}<br>
            <span class="field-label">Rationale</span> ${{action.rationale}}
          </div>
        </div>
      `).join("");
    }}

    function renderContexts() {{
      document.getElementById("contexts").innerHTML = data.contexts.map(ctx => `
        <div class="context-card">
          <div class="context-meta">
            ${{pill(ctx.product_name)}} ${{pill(ctx.sentiment_id)}} ${{pill(ctx.issue_type)}} ${{pill(ctx.confidence)}}
          </div>
          <div class="context-title">${{ctx.context_label}}</div>
          <div class="context-summary">${{ctx.summary}}</div>
          <div class="evidence">
            <span class="field-label">Pair</span> ${{ctx.top_pairs.map(p => `${{p.keyword_1}} ↔ ${{p.keyword_2}} (${{p.npmi.toFixed(2)}})`).join(", ")}}
            <br><span class="field-label">Action</span> ${{ctx.actions.map(a => `${{a.action_type}}/${{a.priority}} - ${{a.title}}`).join(", ")}}
            <br><span class="field-label">Review</span> ${{ctx.evidence_excerpt}}
          </div>
        </div>
      `).join("");
    }}

    function drawNetwork() {{
      const canvas = document.getElementById("network");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      const degree = new Map();
      data.edges.forEach(e => {{
        degree.set(e.source, (degree.get(e.source) || 0) + e.count);
        degree.set(e.target, (degree.get(e.target) || 0) + e.count);
      }});
      const nodes = Array.from(degree.keys());
      const nodeState = new Map();
      nodes.forEach((node, i) => {{
        const angle = i * 2.399963;
        const r = 40 + 12 * Math.sqrt(i);
        nodeState.set(node, {{
          x: w / 2 + Math.cos(angle) * r,
          y: h / 2 + Math.sin(angle) * r,
          vx: 0,
          vy: 0,
        }});
      }});
      for (let step = 0; step < 260; step++) {{
        for (let i = 0; i < nodes.length; i++) {{
          const a = nodeState.get(nodes[i]);
          for (let j = i + 1; j < nodes.length; j++) {{
            const b = nodeState.get(nodes[j]);
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let dist2 = dx * dx + dy * dy + 0.01;
            const force = 1800 / dist2;
            const dist = Math.sqrt(dist2);
            dx /= dist;
            dy /= dist;
            a.vx += dx * force;
            a.vy += dy * force;
            b.vx -= dx * force;
            b.vy -= dy * force;
          }}
        }}
        data.edges.forEach(e => {{
          const a = nodeState.get(e.source), b = nodeState.get(e.target);
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const target = 95 + (1 - e.npmi) * 80;
          const force = (dist - target) * 0.012 * Math.max(0.35, e.npmi);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }});
        nodes.forEach(node => {{
          const n = nodeState.get(node);
          n.vx += (w / 2 - n.x) * 0.004;
          n.vy += (h / 2 - n.y) * 0.004;
          n.vx *= 0.82;
          n.vy *= 0.82;
          n.x = Math.max(46, Math.min(w - 46, n.x + n.vx));
          n.y = Math.max(34, Math.min(h - 46, n.y + n.vy));
        }});
      }}
      data.edges.forEach(e => {{
        const a = nodeState.get(e.source), b = nodeState.get(e.target);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = e.npmi > 0.88 ? "rgba(220,38,38,.62)" : "rgba(100,116,139,.34)";
        ctx.lineWidth = 1 + Math.min(6, e.count / 35);
        ctx.stroke();
      }});
      nodes.forEach(node => {{
        const p = nodeState.get(node);
        const d = degree.get(node);
        const size = 8 + Math.min(20, Math.sqrt(d));
        ctx.beginPath();
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx.fillStyle = d > 100 ? "#2563eb" : "#0891b2";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#111827";
        ctx.font = "13px Arial";
        ctx.textAlign = "center";
        ctx.fillText(node, p.x, p.y + size + 15);
      }});
    }}


    renderStakeholderInsights();
    bars("issueBars", data.label.issue_type_counts, "#7c3aed");
    renderTables();
    renderActionBoard();
    renderContexts();
    drawNetwork();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    print(f"Dashboard written to {args.output}")


if __name__ == "__main__":
    main()
