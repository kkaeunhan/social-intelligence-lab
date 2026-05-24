from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, TypedDict
import html
import re
import unicodedata

VECTOR_DIMS = 768
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
EMBED_MODEL = None
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_MODEL = None

BM25_TOP_K = 100
DENSE_TOP_K = 100
RRF_K = 60
RRF_CANDIDATE_TOP_K = 50
FINAL_TOP_K = 10

BM25_FIELDS = [
    "content^3",
    "title^2",
    "short_summary^2",
    "aspect_keywords.search^2",
    "product_name.search^1.5",
    "item_name^1.2",
]

QUERY_EXPANSION_TERMS = {
    "뽁뽁이": ["뾱뾱이", "에어캡", "완충재", "완충 포장", "보호 포장"],
    "뾱뾱이": ["뽁뽁이", "에어캡", "완충재", "완충 포장"],
    "포장": ["패키징", "박스", "상자", "택배 포장", "완충"],
    "배송": ["택배", "로켓배송", "새벽배송", "도착"],
    "파손": ["찌그러짐", "찍힘", "구겨짐", "뜯김", "흠집"],
    "배터리": ["사용 시간", "방전", "오래감", "하루 종일"],
    "충전": ["충전속도", "고속충전", "완충", "C타입"],
    "카메라": ["사진", "셀카", "화질", "전면 카메라", "후면 카메라", "야간 촬영"],
    "성능": ["속도", "빠릿", "버벅", "렉", "부드러움"],
    "발열": ["뜨거움", "미지근", "열감", "발열관리"],
    "디자인": ["색상", "실물", "외관", "고급", "예쁨"],
    "무게": ["가벼움", "무거움", "그립감", "손목"],
    "화면": ["디스플레이", "주사율", "120Hz", "밝기", "AOD"],
    "가성비": ["가격", "저렴", "비싸", "혜택", "할인"],
    "교환": ["환불", "반품", "AS", "센터", "보증"],
    "반품": ["환불", "교환", "AS", "센터", "보증"],
    "만족": ["좋아요", "추천", "잘 왔", "문제없", "꼼꼼", "안전하게"],
    "불만": ["실망", "비추천", "문제", "후회", "아쉬움", "별로"],
}

ASPECT_QUERY_RULES = {
    "배송/포장": {
        "aspect_terms": ["배송", "포장", "뽁뽁이", "뾱뾱이", "에어캡", "완충", "완충재", "박스", "택배"],
        "positive_terms": ["만족", "좋", "꼼꼼", "안전", "잘 도착", "문제없", "튼튼", "감싸"],
        "negative_terms": ["없이", "없", "파손", "찌그러", "구겨", "뜯겨", "불만", "실망", "반품"],
        "positive_phrases": [
            "뽁뽁이로 잘",
            "뾱뾱이로 잘",
            "뽁뽁이 가득",
            "뾱뾱이 가득",
            "꼼꼼하게 포장",
            "포장 꼼꼼",
            "완충 포장",
            "안전하게 도착",
            "문제없이 도착",
            "잘 도착",
        ],
        "negative_phrases": [
            "뽁뽁이 없이",
            "뾱뾱이 없이",
            "포장 없이",
            "박스 파손",
            "박스 찌그러",
            "박스 구겨",
            "밀봉 뜯",
        ],
    },
    "배터리/충전": {
        "aspect_terms": ["배터리", "충전", "사용 시간", "방전", "완충", "고속충전"],
        "positive_terms": ["오래", "충분", "하루", "든든", "만족", "빠르"],
        "negative_terms": ["짧", "빨리 닳", "방전", "느리", "아쉽", "불만"],
        "positive_phrases": ["배터리 오래", "하루 종일", "충전 빠르", "충분히 사용"],
        "negative_phrases": ["배터리 짧", "빨리 닳", "충전 느리", "금방 방전"],
    },
    "발열": {
        "aspect_terms": ["발열", "열감", "뜨거", "미지근"],
        "positive_terms": ["없", "적", "괜찮", "관리", "만족"],
        "negative_terms": ["심", "뜨겁", "불편", "문제", "아쉽"],
        "positive_phrases": ["발열 없", "발열 적", "뜨겁지 않", "미지근"],
        "negative_phrases": ["발열 심", "너무 뜨겁", "열감 심"],
    },
    "카메라": {
        "aspect_terms": ["카메라", "사진", "셀카", "화질", "촬영", "전면", "후면", "야간"],
        "positive_terms": ["선명", "좋", "잘 나와", "밝", "만족", "예쁘"],
        "negative_terms": ["별로", "흐림", "실망", "안 좋", "아쉽"],
        "positive_phrases": ["사진 잘", "화질 좋", "셀카 잘", "선명하게", "카메라 만족"],
        "negative_phrases": ["카메라 별로", "화질 별로", "사진 흐", "카메라 실망"],
    },
    "성능/속도": {
        "aspect_terms": ["성능", "속도", "렉", "버벅", "빠릿", "스크롤", "부드러움"],
        "positive_terms": ["빠르", "부드럽", "쾌적", "만족", "렉없"],
        "negative_terms": ["느리", "버벅", "렉", "끊", "불편"],
        "positive_phrases": ["속도 빠르", "렉 없", "부드럽게", "성능 좋", "빠릿"],
        "negative_phrases": ["속도 느리", "버벅", "렉 걸", "끊김"],
    },
    "디자인/무게": {
        "aspect_terms": ["디자인", "무게", "색상", "실물", "외관", "그립감", "가벼", "무거"],
        "positive_terms": ["예쁘", "고급", "가볍", "편", "만족", "깔끔"],
        "negative_terms": ["무겁", "별로", "불편", "아쉽", "투박"],
        "positive_phrases": ["색상 예쁘", "실물 예쁘", "가벼워서", "그립감 좋", "디자인 만족"],
        "negative_phrases": ["너무 무겁", "디자인 별로", "그립감 불편"],
    },
    "화면/디스플레이": {
        "aspect_terms": ["화면", "디스플레이", "주사율", "120Hz", "밝기", "AOD", "스크롤"],
        "positive_terms": ["선명", "부드럽", "밝", "좋", "만족", "편"],
        "negative_terms": ["어둡", "끊", "불편", "별로", "아쉽"],
        "positive_phrases": ["화면 선명", "화면 부드럽", "밝기 좋", "눈이 편", "주사율 좋"],
        "negative_phrases": ["화면 어둡", "화면 끊", "디스플레이 별로"],
    },
    "가격/가성비": {
        "aspect_terms": ["가격", "가성비", "저렴", "비싸", "혜택", "할인"],
        "positive_terms": ["좋", "저렴", "만족", "합리", "혜택"],
        "negative_terms": ["비싸", "부담", "아쉽", "불만"],
        "positive_phrases": ["가성비 좋", "가격 좋", "저렴하게", "혜택 좋"],
        "negative_phrases": ["가격 비싸", "가성비 별로", "너무 비싸"],
    },
    "교환/반품/AS": {
        "aspect_terms": ["교환", "반품", "환불", "AS", "센터", "보증"],
        "positive_terms": ["빠르", "친절", "해결", "만족", "가능"],
        "negative_terms": ["불편", "거절", "문제", "실패", "아쉽", "불만"],
        "positive_phrases": ["교환 빠르", "환불 처리", "AS 만족", "해결"],
        "negative_phrases": ["교환 불가", "반품 불편", "환불 안", "AS 불만"],
    },
}

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None
    helpers = None


class SearchOutputItem(TypedDict):
    review_id: str
    score: float
    product_name: str | None
    content: str


class SearchOutput(TypedDict):
    query: str
    results: list[SearchOutputItem]


ISSUE_CATEGORIES = [
    "배송/포장",
    "배터리/충전",
    "발열",
    "카메라",
    "성능/속도",
    "디자인/무게",
    "화면/디스플레이",
    "가격/가성비",
    "교환/반품/AS",
    "기타",
]

ASPECT_SENTIMENT_VALUES = ["positive", "negative", "mixed", "neutral"]


DEFAULT_INDEX_MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "korean_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "review_id": {"type": "keyword"},
            "review_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            "product_name": {
                "type": "keyword",
                "fields": {"search": {"type": "text", "analyzer": "korean_analyzer"}},
            },
            "item_name": {"type": "text", "analyzer": "korean_analyzer"},
            "title": {"type": "text", "analyzer": "korean_analyzer"},
            "content": {"type": "text", "analyzer": "korean_analyzer"},
            "dense_text": {"type": "text", "analyzer": "korean_analyzer"},
            "dense_vector": {
                "type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"
            },
            "review_survey_answers": {
                "type": "nested",
                "properties": {
                    "question": {"type": "keyword"},
                    "answer": {
                        "type": "keyword",
                        "fields": {"search": {"type": "text", "analyzer": "korean_analyzer"}},
                    },
                },
            },
            "rating": {"type": "integer"},
            "helpful_count": {"type": "integer"},
            "helpful_true_count": {"type": "integer"},
            "helpful_false_count": {"type": "integer"},
            "has_image": {"type": "boolean"},
            "has_video": {"type": "boolean"},
            "issue_category": {"type": "keyword"},
            "aspect_keywords": {
                "type": "keyword",
                "fields": {"search": {"type": "text", "analyzer": "korean_analyzer"}},
            },
            "issue_sentiments": {
                "type": "nested",
                "properties": {
                    "issue_category": {"type": "keyword"},
                    "sentiment": {"type": "keyword"},
                    "evidence": {"type": "text", "analyzer": "korean_analyzer"},
                },
            },
            "short_summary": {"type": "text", "analyzer": "korean_analyzer"},
        }
    },
}


@dataclass
class ElasticsearchConfig:
    url: str
    index_name: str
    verify_certs: bool = True


def load_config(env_path: str | None = None) -> ElasticsearchConfig:
    if load_dotenv is not None:
        load_dotenv(env_path)

    return ElasticsearchConfig(
        url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
        index_name=os.getenv("ELASTICSEARCH_INDEX", "reviews"),
        verify_certs=os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true",
    )


def get_client(config: ElasticsearchConfig | None = None) -> Elasticsearch:
    if Elasticsearch is None:
        raise ImportError("Install elasticsearch first: pip install elasticsearch")

    config = config or load_config()
    return Elasticsearch(hosts=[config.url], verify_certs=config.verify_certs)


def embed_text(text: str) -> list[float]:
    if not text:
        return [0.0] * VECTOR_DIMS
    global EMBED_MODEL
    try:
        if EMBED_MODEL is None:
            from sentence_transformers import SentenceTransformer

            EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
        return EMBED_MODEL.encode(text).tolist()
    except Exception as exc:
        raise RuntimeError(
            "Failed to create dense embedding. Check the sentence-transformers/torch "
            "installation before indexing documents."
        ) from exc


def build_dense_text(document: dict[str, Any]) -> str:
    """Build labeled text used as a single dense-retrieval representation."""
    issue_category = document.get("issue_category")
    if isinstance(issue_category, list):
        issue_category_text = ", ".join(str(category) for category in issue_category if category)
    else:
        issue_category_text = issue_category

    fields = [
        ("상품명", document.get("product_name")),
        ("옵션명", document.get("item_name")),
        ("제목", document.get("title")),
        ("본문", document.get("content")),
        ("요약", document.get("short_summary")),
        ("이슈 카테고리", issue_category_text),
    ]

    aspect_keywords = document.get("aspect_keywords")
    if isinstance(aspect_keywords, list):
        aspect_text = ", ".join(str(keyword) for keyword in aspect_keywords if keyword)
    else:
        aspect_text = aspect_keywords
    fields.append(("관점 키워드", aspect_text))

    issue_sentiments = document.get("issue_sentiments") or []
    if isinstance(issue_sentiments, list):
        issue_sentiment_lines = []
        for item in issue_sentiments:
            if not isinstance(item, dict):
                continue
            parts = [
                str(item.get("issue_category") or item.get("aspect", "")).strip(),
                str(item.get("sentiment", "")).strip(),
                str(item.get("evidence", "")).strip(),
            ]
            issue_sentiment_lines.append(" | ".join(part for part in parts if part))
        fields.append(("이슈별 감성", " / ".join(issue_sentiment_lines)))

    survey_answers = document.get("review_survey_answers") or []
    survey_text = " / ".join(
        f"{answer.get('question', '')}: {answer.get('answer', '')}".strip(": ")
        for answer in survey_answers
        if answer.get("question") or answer.get("answer")
    )
    fields.append(("설문 응답", survey_text))

    return "\n".join(
        f"{label}: {str(value).strip()}"
        for label, value in fields
        if value is not None and str(value).strip()
    )


def preprocess_record(record: dict[str, Any]) -> dict[str, Any] | None:
    
    min_content_length = 10

    def clean_text(text: Any) -> str:
        if text is None or str(text).lower() == "nan":
            return ""

        text = unicodedata.normalize("NFKC", html.unescape(str(text)))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", text)
        text = re.sub(r"\b(?:\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})\b", " ", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        text = re.sub(r"(ㅋ)\1{4,}", "ㅋㅋㅋㅋ", text)
        text = re.sub(r"(ㅎ)\1{4,}", "ㅎㅎㅎㅎ", text)
        text = re.sub(r"([!?])\1{2,}", r"\1\1\1", text)
        text = re.sub(r"(\.)\1{2,}", "...", text)
        return re.sub(r"\s+", " ", text).strip()

    def safe_value(val: Any, default: Any = 0):
        if val is None or str(val).lower() == "nan":
            return default
        return val

    def has_enough_signal(text: str) -> bool:
        meaningful_chars = re.findall(r"[0-9A-Za-z가-힣]", text)
        return len(text) >= min_content_length and len(meaningful_chars) >= min_content_length

    review_id = str(record.get("reviewId") or record.get("review_id") or "").strip()
    product_name = clean_text(record.get("product_name", ""))
    title = clean_text(record.get("title"))
    item_name = clean_text(record.get("itemName") or record.get("item_name"))
    content = clean_text(record.get("content"))
    summary = clean_text(record.get("summary") or record.get("short_summary", ""))
    aspect_source = record.get("aspect") or record.get("aspect_keywords") or ""
    if isinstance(aspect_source, list):
        aspect = clean_text(" ".join(str(item) for item in aspect_source))
    else:
        aspect = clean_text(aspect_source)

    if not review_id or not product_name or not content:
        return None
    if not has_enough_signal(content):
        return None

    document = {
        "review_id": review_id,
        "review_at": record.get("reviewAt", 0),
        "product_name": product_name,
        "item_name": item_name,
        "title": title,
        "content": content,
        "review_survey_answers": [
            {
                "question": str(s.get("question", "")),
                "answer": clean_text(s.get("answer", "")),
            }
            for s in (record.get("reviewSurveyAnswers") or [])
            if s.get("question")
        ],
        "rating": int(safe_value(record.get("rating"), 0)),
        "helpful_count": int(safe_value(record.get("helpfulCount"), 0)),
        "helpful_true_count": int(safe_value(record.get("helpfulTrueCount"), 0)),
        "helpful_false_count": int(safe_value(record.get("helpfulFalseCount"), 0)),
        "has_image": len(record.get("attachments") or []) > 0,
        "has_video": len(record.get("videoAttachments") or []) > 0,
    }
    if summary:
        document["short_summary"] = summary
    if aspect:
        document["aspect_keywords"] = [aspect]

    document["dense_text"] = build_dense_text(document)
    document["dense_vector"] = embed_text(document["dense_text"])
    return document


def _empty_llm_metadata() -> dict[str, Any]:
    return {
        "issue_category": ["기타"],
        "aspect_keywords": [],
        "issue_sentiments": [],
        "short_summary": "",
    }


def _build_enrichment_prompt(document: dict[str, Any]) -> str:
    survey_answers = " / ".join(
        f"{answer.get('question', '')}: {answer.get('answer', '')}"
        for answer in document.get("review_survey_answers", [])
        if answer.get("question") or answer.get("answer")
    )

    return f"""
Extract structured metadata from this Korean smartphone e-commerce review.

Allowed issue_category values:
{", ".join(ISSUE_CATEGORIES)}

Rules:
- issue_category must contain exactly 3 issue labels from the allowed list, ordered by importance.
- aspect_keywords must contain exactly 5 short Korean keyword phrases for concrete free-form aspects.
- issue_sentiments must contain 1 to 3 important issue-level sentiment judgments.
- Each issue_sentiments item must use one issue_category label from the allowed list and one sentiment:
  positive, negative, mixed, or neutral.
- issue_sentiments.evidence must be a short Korean phrase grounded in the review.
- short_summary must be one Korean sentence within 80 characters.
- Use only information supported by the review.

Input:
product_name: {document.get("product_name", "")}
rating: {document.get("rating", "")}
title: {document.get("title", "")}
content: {document.get("content", "")}
survey_answers: {survey_answers}
""".strip()


def _normalize_llm_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    issue_category = metadata.get("issue_category", [])
    if isinstance(issue_category, str):
        issue_categories = [issue_category]
    elif isinstance(issue_category, list):
        issue_categories = issue_category
    else:
        issue_categories = []
    normalized_issue_categories = []
    for category in issue_categories:
        category = str(category).strip()
        if category in ISSUE_CATEGORIES and category not in normalized_issue_categories:
            normalized_issue_categories.append(category)
    issue_categories = normalized_issue_categories[:3]
    while len(issue_categories) < 3:
        issue_categories.append("기타")

    aspect_keywords = metadata.get("aspect_keywords", [])
    if not isinstance(aspect_keywords, list):
        aspect_keywords = []
    aspect_keywords = [
        str(keyword).strip()
        for keyword in aspect_keywords
        if str(keyword).strip()
    ][:5]
    while len(aspect_keywords) < 5:
        aspect_keywords.append("기타")

    issue_sentiments = metadata.get("issue_sentiments", [])
    if not isinstance(issue_sentiments, list):
        issue_sentiments = []
    normalized_issue_sentiments = []
    for item in issue_sentiments[:3]:
        if not isinstance(item, dict):
            continue
        item_issue_category = str(item.get("issue_category") or item.get("aspect", "")).strip()
        if item_issue_category not in ISSUE_CATEGORIES:
            item_issue_category = issue_categories[0]
        sentiment = str(item.get("sentiment", "")).strip().lower()
        if sentiment not in ASPECT_SENTIMENT_VALUES:
            sentiment = "neutral"
        evidence = str(item.get("evidence", "")).strip()
        if len(evidence) > 120:
            evidence = evidence[:120].rstrip()
        normalized_issue_sentiments.append(
            {
                "issue_category": item_issue_category,
                "sentiment": sentiment,
                "evidence": evidence,
            }
        )

    short_summary = str(metadata.get("short_summary", "")).strip()
    if len(short_summary) > 80:
        short_summary = short_summary[:80].rstrip()

    return {
        "issue_category": issue_categories,
        "aspect_keywords": aspect_keywords,
        "issue_sentiments": normalized_issue_sentiments,
        "short_summary": short_summary,
    }


def create_openai_enricher(
    model_name: str | None = None,
):
    """Create an OpenAI API based metadata enricher."""
    if load_dotenv is not None:
        load_dotenv()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai to use OpenAI metadata enrichment.") from exc

    model_name = model_name or os.getenv("OPENAI_METADATA_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    metadata_schema = {
        "name": "review_metadata",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issue_category": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string", "enum": ISSUE_CATEGORIES},
                },
                "aspect_keywords": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
                "issue_sentiments": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "issue_category": {"type": "string", "enum": ISSUE_CATEGORIES},
                            "sentiment": {"type": "string", "enum": ASPECT_SENTIMENT_VALUES},
                            "evidence": {
                                "type": "string",
                                "maxLength": 120,
                            },
                        },
                        "required": ["issue_category", "sentiment", "evidence"],
                    },
                },
                "short_summary": {
                    "type": "string",
                    "maxLength": 80,
                },
            },
            "required": [
                "issue_category",
                "aspect_keywords",
                "issue_sentiments",
                "short_summary",
            ],
        },
    }

    def enrich(document: dict[str, Any]) -> dict[str, Any]:
        prompt = _build_enrichment_prompt(document)
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract metadata for Korean e-commerce reviews. "
                        "Return only data that fits the provided JSON schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": metadata_schema,
            },
        )
        content = response.choices[0].message.content or "{}"
        metadata = json.loads(content)
        return _normalize_llm_metadata(metadata)

    return enrich


def enrich_document_with_llm(
    document: dict[str, Any],
    llm_enricher,
) -> dict[str, Any]:
    enriched = dict(document)
    try:
        enriched.update(_normalize_llm_metadata(llm_enricher(document)))
    except Exception:
        enriched.update(_empty_llm_metadata())

    enriched["dense_text"] = build_dense_text(enriched)
    enriched["dense_vector"] = embed_text(enriched["dense_text"])
    return enriched


def build_documents(
    records: Iterable[dict[str, Any]],
    enrich_with_llm: bool = False,
    llm_enricher=None,
) -> list[dict[str, Any]]:
    import re

    def content_fingerprint(content: str) -> str:
        content = content.lower()
        content = re.sub(r"\s+", "", content)
        content = re.sub(r"[^0-9a-z가-힣]", "", content)
        return content

    documents = []
    seen_review_ids = set()
    seen_contents = set()

    for record in records:
        document = preprocess_record(record)
        if document is None:
            continue

        review_id = document["review_id"]
        content_key = content_fingerprint(document["content"])
        if review_id in seen_review_ids or content_key in seen_contents:
            continue

        seen_review_ids.add(review_id)
        seen_contents.add(content_key)
        documents.append(document)

    if enrich_with_llm:
        llm_enricher = llm_enricher or create_openai_enricher()
        documents = [
            enrich_document_with_llm(document, llm_enricher)
            for document in documents
        ]

    return documents


def create_index(
    client: Elasticsearch,
    index_name: str,
    mapping: dict[str, Any] | None = None,
    recreate: bool = False,
) -> None:
    mapping = mapping or DEFAULT_INDEX_MAPPING

    if client.indices.exists(index=index_name):
        if not recreate:
            return
        client.indices.delete(index=index_name)

    client.indices.create(index=index_name, **mapping)


def index_documents(
    documents: Iterable[dict[str, Any]],
    client: Elasticsearch | None = None,
    index_name: str | None = None,
    mapping: dict[str, Any] | None = None,
    recreate: bool = False,
) -> tuple[int, list[Any]]:
    if helpers is None:
        raise ImportError("Install elasticsearch first: pip install elasticsearch")

    config = load_config()
    client = client or get_client(config)
    index_name = index_name or config.index_name

    create_index(client, index_name=index_name, mapping=mapping, recreate=recreate)

    actions = [
        {
            "_index": index_name,
            "_id": document["review_id"],
            "_source": document,
        }
        for document in documents
    ]

    return helpers.bulk(client, actions, refresh="wait_for")


def expand_query(query: str, use_llm: bool | None = None) -> dict[str, Any]:
    """Create lightweight query expansion metadata for retrieval."""
    if load_dotenv is not None:
        load_dotenv()

    target_aspect = _detect_query_aspect(query)
    target_sentiment = _detect_query_sentiment(query)
    aspect_rule = ASPECT_QUERY_RULES.get(target_aspect, {})
    expanded_terms: list[str] = []
    for term, variants in QUERY_EXPANSION_TERMS.items():
        if term in query:
            expanded_terms.extend(variant for variant in variants if variant not in query)
    expanded_terms.extend(aspect_rule.get("aspect_terms", []))
    if target_sentiment == "positive":
        expanded_terms.extend(aspect_rule.get("positive_terms", []))
    elif target_sentiment == "negative":
        expanded_terms.extend(aspect_rule.get("negative_terms", []))

    seen = set()
    expanded_terms = [
        term
        for term in expanded_terms
        if term and not (term in seen or seen.add(term))
    ]

    expansion = {
        "original_query": query,
        "bm25_terms": expanded_terms,
        "dense_queries": [query],
        "must_include": aspect_rule.get("aspect_terms", []),
        "should_include": expanded_terms,
        "should_exclude": _query_exclude_terms(target_sentiment, aspect_rule),
        "target_sentiment": target_sentiment,
        "target_aspect": target_aspect,
        "aspect_terms": aspect_rule.get("aspect_terms", []),
        "positive_terms": aspect_rule.get("positive_terms", []),
        "negative_terms": aspect_rule.get("negative_terms", []),
        "positive_phrases": aspect_rule.get("positive_phrases", []),
        "negative_phrases": aspect_rule.get("negative_phrases", []),
    }

    if use_llm is None:
        use_llm = os.getenv("OPENAI_QUERY_EXPANSION", "false").lower() == "true"
    if use_llm:
        try:
            expansion.update(_expand_query_with_openai(query))
        except Exception:
            pass

    return expansion


def _query_exclude_terms(target_sentiment: str, aspect_rule: dict[str, Any]) -> list[str]:
    if target_sentiment == "positive":
        return aspect_rule.get("negative_terms", []) + aspect_rule.get("negative_phrases", [])
    if target_sentiment == "negative":
        return aspect_rule.get("positive_terms", []) + aspect_rule.get("positive_phrases", [])
    return []


def _detect_query_sentiment(query: str) -> str:
    positive_terms = ["만족", "좋", "추천", "잘", "꼼꼼", "안전", "빠르"]
    negative_terms = ["불만", "실망", "비추천", "파손", "문제", "후회", "아쉽", "별로"]
    positive = any(term in query for term in positive_terms)
    negative = any(term in query for term in negative_terms)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return "unknown"


def _detect_query_aspect(query: str) -> str:
    aspect_terms = {
        "배송/포장": ["배송", "포장", "뽁뽁이", "뾱뾱이", "택배", "박스", "파손"],
        "배터리/충전": ["배터리", "충전", "사용 시간", "방전"],
        "발열": ["발열", "뜨거", "열감"],
        "카메라": ["카메라", "사진", "셀카", "화질", "촬영"],
        "성능/속도": ["성능", "속도", "렉", "버벅", "빠릿", "부드러"],
        "디자인/무게": ["디자인", "무게", "색상", "그립감", "가벼", "무거"],
        "화면/디스플레이": ["화면", "디스플레이", "주사율", "밝기", "AOD"],
        "가격/가성비": ["가격", "가성비", "저렴", "비싸", "할인"],
        "교환/반품/AS": ["교환", "반품", "환불", "AS", "센터", "보증"],
    }
    for aspect, terms in aspect_terms.items():
        if any(term in query for term in terms):
            return aspect
    return "기타"


def _expand_query_with_openai(query: str) -> dict[str, Any]:
    """Optional OpenAI query expansion. Disabled unless OPENAI_QUERY_EXPANSION=true."""
    if load_dotenv is not None:
        load_dotenv()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai to use OpenAI query expansion.") from exc

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("OPENAI_QUERY_EXPANSION_MODEL", "gpt-4.1-mini")
    schema = {
        "name": "query_expansion",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rewritten_query": {"type": "string"},
                "bm25_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "dense_query": {"type": "string"},
                "must_include": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "should_include": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "should_exclude": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "target_sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "unknown"],
                },
                "target_aspect": {"type": "string", "enum": ISSUE_CATEGORIES},
            },
            "required": [
                "rewritten_query",
                "bm25_terms",
                "dense_query",
                "must_include",
                "should_include",
                "should_exclude",
                "target_sentiment",
                "target_aspect",
            ],
        },
    }

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Expand Korean smartphone review search queries without changing "
                    "the user's intent, sentiment, or exclusion conditions."
                ),
            },
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    metadata = json.loads(response.choices[0].message.content or "{}")
    dense_query = str(metadata.get("dense_query") or metadata.get("rewritten_query") or query)
    return {
        "bm25_terms": [
            str(term).strip()
            for term in metadata.get("bm25_terms", [])
            if str(term).strip()
        ],
        "dense_queries": [query, dense_query] if dense_query != query else [query],
        "must_include": [
            str(term).strip()
            for term in metadata.get("must_include", [])
            if str(term).strip()
        ],
        "should_include": [
            str(term).strip()
            for term in metadata.get("should_include", [])
            if str(term).strip()
        ],
        "should_exclude": [
            str(term).strip()
            for term in metadata.get("should_exclude", [])
            if str(term).strip()
        ],
        "target_sentiment": metadata.get("target_sentiment", "unknown"),
        "target_aspect": metadata.get("target_aspect", "기타"),
    }


def _issue_sentiment_filter(expansion: dict[str, Any]) -> dict[str, Any] | None:
    target_aspect = expansion.get("target_aspect")
    target_sentiment = expansion.get("target_sentiment")
    if target_aspect == "기타" or target_sentiment not in ("positive", "negative"):
        return None

    return {
        "nested": {
            "path": "issue_sentiments",
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "issue_sentiments.issue_category": {
                                    "value": target_aspect,
                                }
                            }
                        },
                        {
                            "term": {
                                "issue_sentiments.sentiment": {
                                    "value": target_sentiment,
                                }
                            }
                        },
                    ]
                }
            },
            "score_mode": "none",
            "ignore_unmapped": True,
        }
    }


def build_search_body(query: str, size: int = 10) -> dict[str, Any]:
    expansion = expand_query(query)
    expanded_query = " ".join(expansion.get("bm25_terms", []))
    aspect_query = " ".join(expansion.get("aspect_terms", []))
    sentiment_terms = (
        expansion.get("positive_terms", [])
        if expansion.get("target_sentiment") == "positive"
        else expansion.get("negative_terms", [])
    )
    sentiment_query = " ".join(sentiment_terms)
    target_aspect = expansion.get("target_aspect")

    should_clauses: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": query,
                "fields": BM25_FIELDS,
                "type": "best_fields",
                "operator": "or",
                "boost": 3.0,
            }
        },
        {
            "nested": {
                "path": "review_survey_answers",
                "query": {
                    "match": {
                        "review_survey_answers.answer.search": {
                            "query": query,
                            "boost": 0.6,
                        }
                    }
                },
                "score_mode": "max",
            }
        },
    ]

    if expanded_query:
        should_clauses.append(
            {
                "multi_match": {
                    "query": expanded_query,
                    "fields": BM25_FIELDS,
                    "type": "most_fields",
                    "operator": "or",
                    "boost": 1.2,
                }
            }
        )

    if aspect_query:
        should_clauses.append(
            {
                "multi_match": {
                    "query": aspect_query,
                    "fields": [
                        "content^1.5",
                        "title^1.2",
                        "short_summary^1.5",
                        "aspect_keywords.search^2.5",
                    ],
                    "type": "most_fields",
                    "operator": "or",
                    "boost": 1.6,
                }
            }
        )

    if sentiment_query and target_aspect != "기타":
        should_clauses.append(
            {
                "multi_match": {
                    "query": sentiment_query,
                    "fields": ["content", "title", "short_summary^1.5"],
                    "type": "most_fields",
                    "operator": "or",
                    "boost": 0.8,
                }
            }
        )

    if target_aspect and target_aspect != "기타":
        should_clauses.append({"term": {"issue_category": {"value": target_aspect, "boost": 2.2}}})
        nested_aspect_clauses: list[dict[str, Any]] = [
            {
                "term": {
                    "issue_sentiments.issue_category": {
                        "value": target_aspect,
                        "boost": 3.0,
                    }
                }
            }
        ]
        nested_should_clauses: list[dict[str, Any]] = []
        if expansion.get("target_sentiment") in ("positive", "negative"):
            nested_aspect_clauses.append(
                {
                    "term": {
                        "issue_sentiments.sentiment": {
                            "value": expansion["target_sentiment"],
                            "boost": 3.0,
                        }
                    }
                }
            )
        if aspect_query:
            nested_should_clauses.append(
                {
                    "multi_match": {
                        "query": aspect_query,
                        "fields": [
                            "issue_sentiments.evidence^1.5",
                        ],
                        "type": "most_fields",
                        "operator": "or",
                        "boost": 2.2,
                    }
                }
            )
        if sentiment_query:
            nested_should_clauses.append(
                {
                    "match": {
                        "issue_sentiments.evidence": {
                            "query": sentiment_query,
                            "boost": 1.6,
                        }
                    }
                }
            )
        nested_bool: dict[str, Any] = {"must": nested_aspect_clauses}
        if nested_should_clauses:
            nested_bool["should"] = nested_should_clauses
            nested_bool["minimum_should_match"] = 0
        should_clauses.append(
            {
                "nested": {
                    "path": "issue_sentiments",
                    "query": {"bool": nested_bool},
                    "score_mode": "max",
                    "ignore_unmapped": True,
                }
            }
        )

    phrase_boosts = []
    if expansion.get("target_sentiment") == "positive":
        phrase_boosts = expansion.get("positive_phrases", [])
    elif expansion.get("target_sentiment") == "negative":
        phrase_boosts = expansion.get("negative_phrases", [])
    for phrase in phrase_boosts:
        should_clauses.append(
            {
                "multi_match": {
                    "query": phrase,
                    "fields": ["content^2", "title^1.5", "short_summary^2"],
                    "type": "phrase",
                    "boost": 2.4,
                }
            }
        )

    bool_query: dict[str, Any] = {
        "should": should_clauses,
        "minimum_should_match": 1,
    }
    issue_filter = _issue_sentiment_filter(expansion)
    if issue_filter is not None:
        bool_query["filter"] = [issue_filter]

    return {
        "size": size,
        "_source": {"excludes": ["dense_vector"]},
        "query": {"bool": bool_query},
    }


def _search_bm25(
    client: Elasticsearch,
    index_name: str,
    query: str,
    size: int = BM25_TOP_K,
) -> list[dict[str, Any]]:
    response = client.search(index=index_name, body=build_search_body(query, size=size))
    return _normalize_hits(response, source="bm25")


def _search_dense(
    client: Elasticsearch,
    index_name: str,
    query: str,
    size: int = DENSE_TOP_K,
    expansion: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query_vector = embed_text(query)
    issue_filter = _issue_sentiment_filter(expansion or {})
    body = {
        "size": size,
        "_source": {"excludes": ["dense_vector"]},
        "knn": {
            "field": "dense_vector",
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(size * 5, 300),
        },
    }
    if issue_filter is not None:
        body["knn"]["filter"] = issue_filter
    try:
        response = client.search(index=index_name, body=body)
    except Exception:
        fallback_query = issue_filter or {"match_all": {}}
        response = client.search(
            index=index_name,
            body={
                "size": size,
                "_source": {"excludes": ["dense_vector"]},
                "query": {
                    "script_score": {
                        "query": fallback_query,
                        "script": {
                            "source": (
                                "cosineSimilarity(params.query_vector, "
                                "'dense_vector') + 1.0"
                            ),
                            "params": {"query_vector": query_vector},
                        },
                    }
                },
            },
        )
    return _normalize_hits(response, source="dense")


def _normalize_hits(response: dict[str, Any], source: str) -> list[dict[str, Any]]:
    hits = response.get("hits", {}).get("hits", [])
    normalized = []
    for rank, hit in enumerate(hits, start=1):
        source_doc = hit.get("_source") or {}
        normalized.append(
            {
                "id": str(hit.get("_id") or source_doc.get("review_id", "")),
                "review_id": str(source_doc.get("review_id") or hit.get("_id") or ""),
                "score": float(hit.get("_score") or 0.0),
                "rank": rank,
                "source": source,
                **source_doc,
            }
        )
    return normalized


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    k: int = RRF_K,
    size: int = RRF_CANDIDATE_TOP_K,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for source_name, results in [("bm25", bm25_results), ("dense", dense_results)]:
        for rank, result in enumerate(results, start=1):
            review_id = str(result.get("review_id") or result.get("id") or "")
            if not review_id:
                continue
            candidate = candidates.setdefault(review_id, dict(result))
            candidate["review_id"] = review_id
            candidate[f"{source_name}_rank"] = rank
            candidate[f"{source_name}_score"] = float(result.get("score", 0.0))
            candidate["rrf_score"] = candidate.get("rrf_score", 0.0) + (1.0 / (k + rank))

    return sorted(
        candidates.values(),
        key=lambda item: (
            item.get("rrf_score", 0.0),
            item.get("bm25_score", 0.0),
            item.get("dense_score", 0.0),
        ),
        reverse=True,
    )[:size]


def _build_rerank_text(document: dict[str, Any]) -> str:
    if document.get("dense_text"):
        return str(document["dense_text"])
    return build_dense_text(document)


def _get_reranker():
    global RERANK_MODEL
    if RERANK_MODEL is None:
        from sentence_transformers import CrossEncoder

        model_name = os.getenv("RERANK_MODEL_NAME", RERANK_MODEL_NAME)
        RERANK_MODEL = CrossEncoder(model_name)
    return RERANK_MODEL


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    try:
        reranker = _get_reranker()
        pairs = [(query, _build_rerank_text(candidate)) for candidate in candidates]
        scores = reranker.predict(pairs)
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            scored = dict(candidate)
            scored["rerank_score"] = float(score)
            scored["final_score"] = float(score)
            scored_candidates.append(scored)
        return sorted(
            scored_candidates,
            key=lambda item: item.get("rerank_score", 0.0),
            reverse=True,
        )
    except Exception:
        fallback = []
        for candidate in candidates:
            scored = dict(candidate)
            scored["rerank_score"] = float(scored.get("rrf_score", 0.0))
            scored["final_score"] = float(scored.get("rrf_score", 0.0))
            fallback.append(scored)
        return sorted(
            fallback,
            key=lambda item: item.get("final_score", 0.0),
            reverse=True,
        )


def post_filter_candidates(
    candidates: list[dict[str, Any]],
    size: int = FINAL_TOP_K,
    expansion: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    seen_review_ids = set()
    seen_contents = set()
    target_aspect = (expansion or {}).get("target_aspect")
    target_sentiment = (expansion or {}).get("target_sentiment")

    for candidate in candidates:
        review_id = str(candidate.get("review_id") or candidate.get("id") or "")
        content = str(candidate.get("content") or "").strip()
        if not review_id or not content:
            continue
        if len(re.findall(r"[0-9A-Za-z가-힣]", content)) < 10:
            continue
        if target_aspect != "기타" and target_sentiment in ("positive", "negative"):
            issue_sentiments = candidate.get("issue_sentiments") or []
            has_matching_issue_sentiment = any(
                isinstance(item, dict)
                and item.get("issue_category") == target_aspect
                and item.get("sentiment") == target_sentiment
                for item in issue_sentiments
            )
            if not has_matching_issue_sentiment:
                continue

        content_key = re.sub(r"[^0-9a-z가-힣]", "", content.lower())[:200]
        if review_id in seen_review_ids or content_key in seen_contents:
            continue

        seen_review_ids.add(review_id)
        seen_contents.add(content_key)
        filtered.append(candidate)
        if len(filtered) >= size:
            break

    return filtered


def postprocess_results(
    results: list[dict[str, Any]],
    query: str,
) -> SearchOutput:
    return {
        "query": query,
        "results": [
            {
                "review_id": str(result.get("review_id") or result.get("id") or ""),
                "score": float(
                    result.get("final_score")
                    if result.get("final_score") is not None
                    else result.get("score", 0.0)
                ),
                "product_name": result.get("product_name"),
                "content": str(result.get("content") or ""),
            }
            for result in results
        ],
    }


def validate_search_output(output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    if not isinstance(output.get("query"), str):
        return False
    if not isinstance(output.get("results"), list):
        return False

    for item in output["results"]:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("review_id"), str):
            return False
        if not isinstance(item.get("score"), (int, float)):
            return False
        if item.get("product_name") is not None and not isinstance(
            item.get("product_name"),
            str,
        ):
            return False
        if not isinstance(item.get("content"), str):
            return False

    return True


def search(
    query: str,
    client: Elasticsearch | None = None,
    index_name: str | None = None,
    size: int = 10,
) -> SearchOutput:
    config = load_config()
    client = client or get_client(config)
    index_name = index_name or config.index_name

    expansion = expand_query(query)
    bm25_results = _search_bm25(
        client=client,
        index_name=index_name,
        query=query,
        size=BM25_TOP_K,
    )

    dense_by_id: dict[str, dict[str, Any]] = {}
    for dense_query in expansion.get("dense_queries", [query]):
        for result in _search_dense(
            client=client,
            index_name=index_name,
            query=str(dense_query),
            size=DENSE_TOP_K,
            expansion=expansion,
        ):
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
    reranked = rerank_candidates(query=query, candidates=rrf_candidates)
    filtered = post_filter_candidates(
        candidates=reranked,
        size=size or FINAL_TOP_K,
        expansion=expansion,
    )
    output = postprocess_results(filtered, query=query)

    if not validate_search_output(output):
        raise ValueError("search output does not match SearchOutput schema")
    return output
