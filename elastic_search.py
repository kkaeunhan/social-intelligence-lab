from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, TypedDict

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


# TODO: Update this mapping to match the documents you will index.
DEFAULT_INDEX_MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "korean_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "review_id": {"type": "keyword"},
            "review_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},

            "product_name": {"type": "keyword", "fields": {"search": {"type": "text", "analyzer": "korean_analyzer"}}},
            "item_name": {"type": "text", "analyzer": "korean_analyzer"},
            "title": {"type": "text", "analyzer": "korean_analyzer"},
            "content": {"type": "text", "analyzer": "korean_analyzer"},
            "review_survey_answers": {
                "type": "nested",
                "properties": {
                    "question": {"type": "keyword"},
                    "answer": {"type": "keyword", "fields": {"search": {"type": "text", "analyzer": "korean_analyzer"}}}
                }
            },

            "rating": {"type": "integer"},
            "helpful_count": {"type": "integer"},
            "helpful_true_count": {"type": "integer"},
            "helpful_false_count": {"type": "integer"},
            "has_image": {"type": "boolean"},
            "has_video": {"type": "boolean"}
        }
    }
}


@dataclass
class ElasticsearchConfig:
    url: str
    index_name: str
    verify_certs: bool = True


def load_config(env_path: str | None = None) -> ElasticsearchConfig:
    """Load Elasticsearch settings from .env or environment variables.

    Expected variables:
    - ELASTICSEARCH_URL
    - ELASTICSEARCH_INDEX
    - ELASTICSEARCH_VERIFY_CERTS
    """
    if load_dotenv is not None:
        load_dotenv(env_path)

    return ElasticsearchConfig(
        url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
        index_name=os.getenv("ELASTICSEARCH_INDEX", "reviews"),
        verify_certs=os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true",
    )


def get_client(config: ElasticsearchConfig | None = None) -> Elasticsearch:
    """Create an Elasticsearch client from config."""
    if Elasticsearch is None:
        raise ImportError("Install elasticsearch first: pip install elasticsearch")

    config = config or load_config()
    client_options = {
        "hosts": [config.url],
        "verify_certs": config.verify_certs,
    }

    return Elasticsearch(**client_options)

def preprocess_record(record: dict[str, Any]) -> dict[str, Any] | None:
    import re
    import html
    import unicodedata

    min_content_length = 10

    def clean_text(text: Any) -> str:
        if text is None or str(text).lower() == "nan":
            return ""
        
        text = unicodedata.normalize('NFKC', html.unescape(str(text)))
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

    doc = {
        "review_id": review_id,
        "review_at": record.get("reviewAt", 0),

        "product_name": product_name,
        "item_name": clean_text(record.get("itemName")),
        "title": title,
        "content": content,

        "review_survey_answers": [
            {
                "question": str(s.get("question", "")),
                "answer": clean_text(s.get("answer", ""))
            }
            for s in (record.get("reviewSurveyAnswers") or [])
            if s.get("question")
        ],

        "rating": int(safe_value(record.get("rating"), 0)),
        "helpful_count": int(safe_value(record.get("helpfulCount"), 0)),
        "helpful_true_count": int(safe_value(record.get("helpfulTrueCount"), 0)),
        "helpful_false_count": int(safe_value(record.get("helpfulFalseCount"), 0)),
        "has_image": len(record.get("attachments") or []) > 0,
        "has_video": len(record.get("videoAttachments") or []) > 0
    }

    return doc


def build_documents(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build preprocessed documents before calling index_documents."""
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

    return documents


def create_index(
    client: Elasticsearch,
    index_name: str,
    mapping: dict[str, Any] | None = None,
    recreate: bool = False,
) -> None:
    """Create an index with a supplied mapping.

    Set recreate=True to delete and recreate the index.
    """
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
    """Bulk index documents that are already preprocessed."""
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
    """Build an Elasticsearch query body from a user query."""
    # TODO: Implement your retrieval strategy.
    raise NotImplementedError("build_search_body is not implemented yet.")


def postprocess_results(
    results: list[dict[str, Any]],
    query: str,
) -> SearchOutput:
    """
    Post-process search results before returning them.
    (e.g. reranking, filtering, transforming)
    """
    # TODO: Rerank, filter, or transform search results.
    raise NotImplementedError("postprocess_results is not implemented yet.")


def validate_search_output(output: Any) -> bool:
    """Check whether search output follows the required schema."""
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
    """Run the full retrieval pipeline and return validated search output."""
    # TODO: Implement your retrieval pipeline.
    raise NotImplementedError("search is not implemented yet.")
