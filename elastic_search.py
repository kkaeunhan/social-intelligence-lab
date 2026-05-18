from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, TypedDict
import html
import re
import unicodedata

from sentence_transformers import SentenceTransformer
EMBED_MODEL = SentenceTransformer('jhgan/ko-sroberta-multitask')
VECTOR_DIMS = 768

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
            "content_vector": {
                "type": "dense_vector",
                "dims": 768,            
                "index": True,           
                "similarity": "cosine"   
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
    content = clean_text(record.get("content"))

    if not review_id or not product_name or not content:
        return None
    if not has_enough_signal(content):
        return None
    
    try:
        content_vector = EMBED_MODEL.encode(content).tolist()
    except Exception as e:
        print(f"벡터 변환 에러 (review_id: {review_id}): {e}")
        content_vector = [0.0] * VECTOR_DIMS

    return {
        "review_id": review_id,
        "review_at": record.get("reviewAt", 0),
        "product_name": product_name,
        "item_name": clean_text(record.get("itemName")),
        "title": title,
        "content": content,
        "content_vector": content_vector,
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


def _empty_llm_metadata() -> dict[str, Any]:
    return {
        "issue_category": "기타",
        "aspect_keywords": [],
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
- issue_category must be exactly one value from the allowed list.
- aspect_keywords must contain exactly 5 short Korean keyword phrases.
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
    issue_category = str(metadata.get("issue_category", "")).strip()
    if issue_category not in ISSUE_CATEGORIES:
        issue_category = "기타"

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

    short_summary = str(metadata.get("short_summary", "")).strip()
    if len(short_summary) > 80:
        short_summary = short_summary[:80].rstrip()

    return {
        "issue_category": issue_category,
        "aspect_keywords": aspect_keywords,
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
                    "type": "string",
                    "enum": ISSUE_CATEGORIES,
                },
                "aspect_keywords": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
                "short_summary": {
                    "type": "string",
                    "maxLength": 80,
                },
            },
            "required": [
                "issue_category",
                "aspect_keywords",
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


def build_search_body(query: str, size: int = 10) -> dict[str, Any]:
    raise NotImplementedError("build_search_body is not implemented yet.")


def postprocess_results(
    results: list[dict[str, Any]],
    query: str,
) -> SearchOutput:
    raise NotImplementedError("postprocess_results is not implemented yet.")


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
    raise NotImplementedError("search is not implemented yet.")
