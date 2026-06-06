from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT_DIR / "task2.2_network"
DATA_DIR = TASK_DIR / "data"

DEFAULT_CLEANED_REVIEWS = DATA_DIR / "cleaned_reviews.jsonl"
DEFAULT_CO_OCCURRENCE_EDGES = DATA_DIR / "co_occurrence_edges.jsonl"
DEFAULT_CANDIDATES = DATA_DIR / "context_insight_candidates.jsonl"
DEFAULT_CLUSTERS = DATA_DIR / "context_candidate_clusters.jsonl"
DEFAULT_SUMMARY = DATA_DIR / "context_candidate_summary.json"
DEFAULT_REPORT = DATA_DIR / "context_insight_candidates.md"

TARGET_SCOPES = {"product_sentiment", "product", "sentiment", "global"}
SENTIMENT_ORDER = {"negative": 0, "positive": 1, "neutral": 2, None: 3}
ACTION_BY_SENTIMENT = {
    "negative": "product_improvement_or_cs",
    "positive": "marketing_message",
    "neutral": "monitoring_or_detail_page_copy",
}

THEME_RULES = [
    (
        "배송/포장 리스크",
        {"배송", "포장", "박스", "뽁뽁이", "완충", "상자", "찌그러", "파손", "밀봉", "새벽", "로켓"},
    ),
    (
        "불량/교환/환불",
        {"불량", "교환", "환불", "반품", "하자", "결함", "고장", "문제", "센터", "보상"},
    ),
    (
        "가격/가성비",
        {"가격", "가성비", "비싸", "저렴", "할인", "쿠폰", "혜택", "카드", "사전", "예약"},
    ),
    (
        "카메라/화질",
        {"카메라", "사진", "영상", "화질", "줌", "야간", "렌즈", "색감", "선명"},
    ),
    (
        "배터리/충전",
        {"배터리", "충전", "사용 시간", "고속", "오래", "짧", "닳", "발열"},
    ),
    (
        "성능/속도",
        {"성능", "속도", "버벅", "cpu", "gpu", "칩", "스냅드래곤", "엑시노스", "램", "발열"},
    ),
    (
        "디자인/색상/마감",
        {"디자인", "색상", "색깔", "컬러", "마감", "고급", "예쁘", "바이올렛", "코발트"},
    ),
    (
        "무게/휴대성/그립",
        {"무게", "무겁", "가볍", "손목", "그립", "크기", "휴대", "한손"},
    ),
    (
        "화면/디스플레이",
        {"화면", "디스플레이", "밝기", "주사율", "hz", "amoled", "베젤", "스크린"},
    ),
    (
        "데이터 이전/설정",
        {"스마트", "스위치", "이전", "백업", "설정", "연동", "통화", "녹음"},
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare pre-LLM context insight candidates from PMI/NPMI graph edges."
    )
    parser.add_argument("--cleaned-reviews", type=Path, default=DEFAULT_CLEANED_REVIEWS)
    parser.add_argument("--co-occurrence-edges", type=Path, default=DEFAULT_CO_OCCURRENCE_EDGES)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cluster-output", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--min-npmi", type=float, default=0.2)
    parser.add_argument("--top-per-group", type=int, default=12)
    parser.add_argument("--top-product-sentiment", type=int, default=8)
    parser.add_argument("--sample-reviews", type=int, default=3)
    parser.add_argument("--excerpt-chars", type=int, default=180)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for review in iter_jsonl(path):
        reviews[review["review_id"]] = review
    return reviews


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_candidates(review: dict[str, Any]) -> list[str]:
    parts = [
        str(review.get("title") or ""),
        str(review.get("content") or ""),
        str(review.get("survey_text") or ""),
    ]
    text = clean_space(" ".join(part for part in parts if part))
    chunks = re.split(r"(?<=[.!?。！？])\s+|[\\n\r]+| / ", text)
    return [clean_space(chunk) for chunk in chunks if clean_space(chunk)]


def best_excerpt(
    review: dict[str, Any],
    keywords: list[str],
    *,
    max_chars: int,
) -> str:
    chunks = sentence_candidates(review)
    if not chunks:
        return clean_space(str(review.get("analysis_text") or ""))[:max_chars]

    lowered_keywords = [keyword.lower() for keyword in keywords if keyword]

    def score(chunk: str) -> tuple[int, int]:
        lower = chunk.lower()
        hits = sum(1 for keyword in lowered_keywords if keyword in lower)
        return hits, -len(chunk)

    best = max(chunks, key=score)
    if len(best) <= max_chars:
        return best

    lower = best.lower()
    hit_positions = [
        lower.find(keyword)
        for keyword in lowered_keywords
        if keyword and lower.find(keyword) >= 0
    ]
    center = min(hit_positions) if hit_positions else 0
    start = max(0, center - max_chars // 3)
    end = min(len(best), start + max_chars)
    start = max(0, end - max_chars)
    excerpt = best[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(best):
        excerpt += "..."
    return excerpt


def infer_theme(keyword_1: str, keyword_2: str) -> tuple[str, list[str]]:
    joined = f"{keyword_1} {keyword_2}".lower()
    matches: list[str] = []
    for theme, clues in THEME_RULES:
        if any(clue.lower() in joined for clue in clues):
            matches.append(theme)
    if matches:
        return matches[0], matches
    return "기타 제품 인식", []


def evidence_strength(count: int, npmi: float) -> str:
    if count >= 20 and npmi >= 0.45:
        return "strong"
    if count >= 8 and npmi >= 0.3:
        return "medium"
    return "exploratory"


def significance_score(edge: dict[str, Any]) -> float:
    count = int(edge.get("count") or edge.get("co_count") or 0)
    npmi = float(edge.get("npmi") or 0)
    pmi = float(edge.get("pmi") or 0)
    lift = float(edge.get("lift") or 0)
    review_count = int(edge.get("review_count") or 1)
    support = math.log1p(count)
    coverage = count / max(review_count, 1)
    lift_bonus = min(math.log1p(max(lift, 0)) / 5, 1.0)
    pmi_bonus = min(max(pmi, 0) / 10, 1.0)
    return round((npmi * 0.62 + pmi_bonus * 0.18 + lift_bonus * 0.1 + coverage * 0.1) * support, 6)


def business_action_hint(sentiment_id: str | None, theme: str) -> str:
    if sentiment_id == "negative":
        if theme in {"배송/포장 리스크", "불량/교환/환불"}:
            return "CS 응대 스크립트와 배송/검수 개선 과제로 우선 검토"
        if theme in {"배터리/충전", "성능/속도"}:
            return "상세페이지 기대치 조정, 사용 가이드, 제품 개선 이슈로 분리"
        return "부정 리뷰 원문 확인 후 불만 원인과 반복 패턴 점검"
    if sentiment_id == "positive":
        if theme in {"카메라/화질", "디자인/색상/마감", "성능/속도"}:
            return "상품 상세와 광고 소재에서 강점 메시지로 활용"
        return "긍정 인식 근거를 상세페이지/리뷰 큐레이션에 반영"
    return "중립/혼합 맥락으로 모니터링하고 제품별 차이를 비교"


def build_candidate(edge: dict[str, Any], reviews: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    keyword_1 = edge.get("keyword_1") or edge.get("keyword_a")
    keyword_2 = edge.get("keyword_2") or edge.get("keyword_b")
    sentiment_id = edge.get("sentiment_id")
    theme, theme_matches = infer_theme(keyword_1, keyword_2)
    sample_ids = list(edge.get("sample_review_ids") or [])[: args.sample_reviews]
    evidence_reviews = []
    for review_id in sample_ids:
        review = reviews.get(str(review_id))
        if not review:
            continue
        evidence_reviews.append(
            {
                "review_id": review["review_id"],
                "product_name": review.get("product_name"),
                "rating": review.get("rating"),
                "sentiment_id": review.get("sentiment_id"),
                "helpful_count": review.get("helpful_count"),
                "review_at": review.get("review_at"),
                "excerpt": best_excerpt(
                    review,
                    [keyword_1, keyword_2],
                    max_chars=args.excerpt_chars,
                ),
            }
        )

    count = int(edge.get("count") or edge.get("co_count") or 0)
    npmi = float(edge.get("npmi") or 0)
    score = significance_score(edge)
    return {
        "candidate_id": "|".join(
            [
                str(edge.get("scope")),
                str(edge.get("product_name") or "all_products"),
                str(sentiment_id or "all_sentiments"),
                str(edge.get("pair_key") or f"{keyword_1}||{keyword_2}"),
            ]
        ),
        "scope": edge.get("scope"),
        "product_name": edge.get("product_name"),
        "sentiment_id": sentiment_id,
        "keyword_1": keyword_1,
        "keyword_2": keyword_2,
        "source_keyword_id": edge.get("source_keyword_id") or edge.get("keyword_a_id"),
        "target_keyword_id": edge.get("target_keyword_id") or edge.get("keyword_b_id"),
        "theme_hint": theme,
        "theme_rule_matches": theme_matches,
        "business_action_type_hint": ACTION_BY_SENTIMENT.get(sentiment_id, "comparative_analysis"),
        "business_action_hint": business_action_hint(sentiment_id, theme),
        "count": count,
        "review_count": edge.get("review_count"),
        "pmi": edge.get("pmi"),
        "npmi": npmi,
        "lift": edge.get("lift"),
        "confidence": edge.get("confidence"),
        "significance_score": score,
        "evidence_strength": evidence_strength(count, npmi),
        "sample_review_ids": sample_ids,
        "evidence_reviews": evidence_reviews,
        "llm_ready": bool(evidence_reviews),
        "llm_instruction_seed": (
            "이 후보의 keyword pair, 제품, 감성, 원문 근거를 바탕으로 고객 인식 맥락, "
            "비즈니스 액션, 신뢰도를 과장 없이 도출한다."
        ),
    }


def group_key(edge: dict[str, Any]) -> tuple[str, str | None, str | None]:
    return edge.get("scope"), edge.get("product_name"), edge.get("sentiment_id")


def select_candidates(
    edges: Iterable[dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge.get("scope") not in TARGET_SCOPES:
            continue
        count = int(edge.get("count") or edge.get("co_count") or 0)
        npmi = float(edge.get("npmi") or 0)
        if count < args.min_count or npmi < args.min_npmi:
            continue
        grouped[group_key(edge)].append(edge)

    selected: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        scope, _product_name, _sentiment_id = key
        limit = args.top_product_sentiment if scope == "product_sentiment" else args.top_per_group
        rows.sort(
            key=lambda row: (
                -significance_score(row),
                -float(row.get("npmi") or 0),
                -int(row.get("count") or row.get("co_count") or 0),
                str(row.get("keyword_1") or row.get("keyword_a")),
                str(row.get("keyword_2") or row.get("keyword_b")),
            )
        )
        selected.extend(build_candidate(row, reviews, args) for row in rows[:limit])

    selected.sort(
        key=lambda row: (
            str(row["scope"]),
            str(row.get("product_name") or ""),
            SENTIMENT_ORDER.get(row.get("sentiment_id"), 9),
            -row["significance_score"],
            -row["count"],
        )
    )
    return selected


def summarize(candidates: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_scope = Counter(row["scope"] for row in candidates)
    by_product = Counter(row.get("product_name") or "all_products" for row in candidates)
    by_sentiment = Counter(row.get("sentiment_id") or "all_sentiments" for row in candidates)
    by_theme = Counter(row["theme_hint"] for row in candidates)
    llm_ready = sum(1 for row in candidates if row["llm_ready"])
    return {
        "candidate_count": len(candidates),
        "llm_ready_candidate_count": llm_ready,
        "filters": {
            "min_count": args.min_count,
            "min_npmi": args.min_npmi,
            "top_per_group": args.top_per_group,
            "top_product_sentiment": args.top_product_sentiment,
            "sample_reviews": args.sample_reviews,
        },
        "candidate_counts_by_scope": dict(sorted(by_scope.items())),
        "candidate_counts_by_product": dict(sorted(by_product.items())),
        "candidate_counts_by_sentiment": dict(sorted(by_sentiment.items())),
        "candidate_counts_by_theme": dict(by_theme.most_common()),
        "top_candidates": [
            {
                "scope": row["scope"],
                "product_name": row.get("product_name"),
                "sentiment_id": row.get("sentiment_id"),
                "keyword_1": row["keyword_1"],
                "keyword_2": row["keyword_2"],
                "theme_hint": row["theme_hint"],
                "count": row["count"],
                "npmi": row["npmi"],
                "significance_score": row["significance_score"],
            }
            for row in sorted(
                candidates,
                key=lambda row: (-row["significance_score"], -row["count"]),
            )[:20]
        ],
    }


def build_clusters(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None, str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["scope"] != "product_sentiment":
            continue
        key = (
            candidate["scope"],
            candidate.get("product_name"),
            candidate.get("sentiment_id"),
            candidate["theme_hint"],
        )
        grouped[key].append(candidate)

    clusters: list[dict[str, Any]] = []
    for (scope, product_name, sentiment_id, theme), rows in grouped.items():
        rows.sort(key=lambda row: (-row["significance_score"], -row["count"]))
        top_pairs = [
            {
                "keyword_1": row["keyword_1"],
                "keyword_2": row["keyword_2"],
                "count": row["count"],
                "npmi": row["npmi"],
                "pmi": row["pmi"],
                "significance_score": row["significance_score"],
                "evidence_strength": row["evidence_strength"],
            }
            for row in rows[:6]
        ]

        evidence_reviews: list[dict[str, Any]] = []
        seen_review_ids: set[str] = set()
        for row in rows:
            for review in row["evidence_reviews"]:
                review_id = str(review["review_id"])
                if review_id in seen_review_ids:
                    continue
                evidence_reviews.append(review)
                seen_review_ids.add(review_id)
                if len(evidence_reviews) >= 5:
                    break
            if len(evidence_reviews) >= 5:
                break

        counts = [int(row["count"]) for row in rows]
        scores = [float(row["significance_score"]) for row in rows]
        npmies = [float(row["npmi"]) for row in rows]
        clusters.append(
            {
                "cluster_id": "|".join([scope, str(product_name), str(sentiment_id), theme]),
                "scope": scope,
                "product_name": product_name,
                "sentiment_id": sentiment_id,
                "theme_hint": theme,
                "candidate_count": len(rows),
                "total_pair_count": sum(counts),
                "max_pair_count": max(counts),
                "avg_npmi": round(sum(npmies) / len(npmies), 6),
                "max_npmi": max(npmies),
                "max_significance_score": max(scores),
                "business_action_type_hint": ACTION_BY_SENTIMENT.get(
                    sentiment_id, "comparative_analysis"
                ),
                "business_action_hint": business_action_hint(sentiment_id, theme),
                "top_pairs": top_pairs,
                "evidence_reviews": evidence_reviews,
                "llm_ready": bool(top_pairs and evidence_reviews),
                "llm_task": (
                    "제품/감성/테마별 연관어쌍과 리뷰 근거를 종합해 context_label, "
                    "customer_perception, business_action, confidence를 생성한다."
                ),
            }
        )

    clusters.sort(
        key=lambda row: (
            str(row.get("product_name") or ""),
            SENTIMENT_ORDER.get(row.get("sentiment_id"), 9),
            -row["max_significance_score"],
            row["theme_hint"],
        )
    )
    return clusters


def render_report(candidates: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Step 6 Pre-LLM Context Insight Candidates",
        "",
        "LLM 해석 전에 PMI/NPMI 기반 연관어쌍을 제품/감성/테마 단위로 선별한 후보 목록입니다.",
        "",
        "## Summary",
        "",
        f"- Candidate count: {summary['candidate_count']}",
        f"- LLM-ready candidate count: {summary['llm_ready_candidate_count']}",
        f"- Filters: min_count={summary['filters']['min_count']}, min_npmi={summary['filters']['min_npmi']}",
        "",
        "## Theme Counts",
        "",
    ]
    for theme, count in summary["candidate_counts_by_theme"].items():
        lines.append(f"- {theme}: {count}")

    lines.extend(["", "## Product-Sentiment Candidates", ""])
    product_sentiment_rows = [
        row for row in candidates if row["scope"] == "product_sentiment"
    ]
    current_group: tuple[str | None, str | None] | None = None
    for row in product_sentiment_rows:
        group = (row.get("product_name"), row.get("sentiment_id"))
        if group != current_group:
            current_group = group
            lines.extend(
                [
                    "",
                    f"### {group[0]} / {group[1]}",
                    "",
                ]
            )
        lines.append(
            "- "
            f"{row['keyword_1']} + {row['keyword_2']} "
            f"(theme={row['theme_hint']}, count={row['count']}, "
            f"npmi={row['npmi']}, score={row['significance_score']}, "
            f"strength={row['evidence_strength']})"
        )
        if row["evidence_reviews"]:
            excerpt = row["evidence_reviews"][0]["excerpt"]
            lines.append(f"  - evidence: {excerpt}")

    lines.extend(["", "## Global/Comparative Candidates", ""])
    for row in candidates:
        if row["scope"] == "product_sentiment":
            continue
        label = row["scope"]
        if row.get("product_name"):
            label += f"/{row['product_name']}"
        if row.get("sentiment_id"):
            label += f"/{row['sentiment_id']}"
        lines.append(
            "- "
            f"[{label}] {row['keyword_1']} + {row['keyword_2']} "
            f"(theme={row['theme_hint']}, count={row['count']}, npmi={row['npmi']})"
        )

    lines.append("")
    return "\n".join(lines)


def add_cluster_summary(summary: dict[str, Any], clusters: list[dict[str, Any]]) -> None:
    by_product = Counter(row.get("product_name") or "all_products" for row in clusters)
    by_sentiment = Counter(row.get("sentiment_id") or "all_sentiments" for row in clusters)
    by_theme = Counter(row["theme_hint"] for row in clusters)
    summary["cluster_count"] = len(clusters)
    summary["llm_ready_cluster_count"] = sum(1 for row in clusters if row["llm_ready"])
    summary["cluster_counts_by_product"] = dict(sorted(by_product.items()))
    summary["cluster_counts_by_sentiment"] = dict(sorted(by_sentiment.items()))
    summary["cluster_counts_by_theme"] = dict(by_theme.most_common())
    summary["top_clusters"] = [
        {
            "product_name": row["product_name"],
            "sentiment_id": row["sentiment_id"],
            "theme_hint": row["theme_hint"],
            "candidate_count": row["candidate_count"],
            "total_pair_count": row["total_pair_count"],
            "max_npmi": row["max_npmi"],
            "max_significance_score": row["max_significance_score"],
        }
        for row in sorted(
            clusters,
            key=lambda row: (-row["max_significance_score"], -row["total_pair_count"]),
        )[:20]
    ]


def main() -> None:
    args = parse_args()
    reviews = load_reviews(args.cleaned_reviews)
    edges = list(iter_jsonl(args.co_occurrence_edges))
    candidates = select_candidates(edges, reviews, args)
    clusters = build_clusters(candidates)
    summary = summarize(candidates, args)
    add_cluster_summary(summary, clusters)

    write_jsonl(args.candidate_output, candidates)
    write_jsonl(args.cluster_output, clusters)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    args.report_output.write_text(render_report(candidates, summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
