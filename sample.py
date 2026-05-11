from __future__ import annotations

import argparse
import random
from typing import Any

import elastic_search
import sentiment


LABELS = [0, 1, 2, 3]

DUMMY_RECORDS = [
    {
        "review_id": "sample-001",
        "product_name": "iphone_17",
        "rating": 5,
        "content": "화면이 선명하고 배터리 오래가서 만족합니다.",
        "label": 3,
    },
    {
        "review_id": "sample-002",
        "product_name": "iphone_17",
        "rating": 1,
        "content": "포장이 찢어져서 왔고 제품 상태도 별로였습니다.",
        "label": 0,
    },
    {
        "review_id": "sample-003",
        "product_name": "galaxy_s26",
        "rating": 4,
        "content": "가격은 비싸지만 성능은 괜찮습니다.",
        "label": 2,
    },
]

SAMPLE_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "review_id": {"type": "keyword"},
            "product_name": {"type": "keyword"},
            "rating": {"type": "integer"},
            "content": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"raw": {"type": "keyword"}},
            },
            "label": {"type": "integer"},
        }
    }
}


def predict_sentiment(content: str) -> int:
    """Return a random sentiment label."""
    return random.choice(LABELS)


def preprocess_content(content: str) -> str:
    """Sample preprocessing: remove leading and trailing whitespace."""
    return content.strip()


def run_sentiment_sample(
    file_path: str = "si_dataset/train_review_data.json",
    output_dir: str = "output",
) -> dict[str, Any]:
    """Run random sentiment prediction and evaluate it."""
    result = sentiment.run_sentiment_analysis(
        file_path=file_path,
        output_dir=output_dir,
        preprocess_func=preprocess_content,
        predict_func=predict_sentiment,
    )
    print(f"random baseline QWK: {result['score']:.4f}")
    print("prediction sample:", result["predictions"][:3])
    print("saved to:", result["output_path"])
    return result


def sample_preprocess_record(record: dict[str, Any]) -> dict[str, Any]:
    """Sample implementation for elastic_search.preprocess_record."""
    return {
        "review_id": str(record["review_id"]),
        "product_name": record["product_name"],
        "rating": int(record["rating"]),
        "content": record["content"],
        "label": record.get("label"),
    }


def build_search_body(query: str, size: int = 1) -> dict[str, Any]:
    """Sample implementation for elastic_search.build_search_body."""
    return {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content^2", "product_name"],
                        }
                    },
                    {"wildcard": {"content.raw": f"*{query}*"}},
                    {"wildcard": {"product_name": f"*{query}*"}},
                ],
                "minimum_should_match": 1,
            },
        },
    }


def postprocess_results(
    results: list[dict[str, Any]],
    query: str,
) -> elastic_search.SearchOutput:
    """Sample implementation for elastic_search.postprocess_results."""
    return {
        "query": query,
        "results": [
            {
                "review_id": str(result["id"]),
                "score": float(result["score"]),
                "product_name": result.get("product_name"),
                "content": result.get("content", ""),
            }
            for result in results
        ],
    }


def search(
    query: str,
    client: elastic_search.Elasticsearch | None = None,
    index_name: str | None = None,
    size: int = 10,
) -> elastic_search.SearchOutput:
    """Sample implementation for the full retrieval pipeline."""
    config = elastic_search.load_config()
    client = client or elastic_search.get_client(config)
    index_name = index_name or config.index_name
    body = build_search_body(query=query, size=size)

    response = client.search(index=index_name, body=body)

    results = [
        {
            "id": hit["_id"],
            "score": hit["_score"],
            **hit["_source"],
        }
        for hit in response["hits"]["hits"]
    ]
    output = postprocess_results(results, query)
    if not elastic_search.validate_search_output(output):
        raise ValueError("Search output does not match the required schema.")

    return output


def run_index_sample(index_name: str = "sample_reviews") -> tuple[int, list[Any]]:
    """Index dummy records into Elasticsearch."""
    documents = [sample_preprocess_record(record) for record in DUMMY_RECORDS]

    client = elastic_search.get_client()
    print(f"recreating index: {index_name}")
    indexed_count, errors = elastic_search.index_documents(
        documents=documents,
        client=client,
        index_name=index_name,
        mapping=SAMPLE_INDEX_MAPPING,
        recreate=True,
    )
    client.indices.refresh(index=index_name)
    print(f"indexed documents: {indexed_count}, errors: {errors}")
    return indexed_count, errors


def run_search_sample(
    index_name: str = "sample_reviews",
    query: str = "anything",
    size: int = 1,
) -> elastic_search.SearchOutput:
    """Search documents from an Elasticsearch index."""
    output = search(query=query, index_name=index_name, size=size)

    print("search query:", query)
    print("search results:", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sample sentiment evaluation and Elasticsearch indexing/search."
    )
    parser.add_argument(
        "task",
        choices=["sentiment", "index", "search"],
        nargs="?",
        default="sentiment",
        help=(
            "sentiment: run random sentiment evaluation, "
            "index: index dummy data, "
            "search: search indexed documents"
        ),
    )
    parser.add_argument(
        "query_text",
        nargs="?",
        default=None,
        help='Search query for the search task. Example: python sample.py search "battery"',
    )
    parser.add_argument(
        "--data-path",
        default="si_dataset/train_review_data.json",
        help="Review data path for the sentiment sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for sentiment prediction output JSON files.",
    )
    parser.add_argument(
        "--index-name",
        default="sample_reviews",
        help="Elasticsearch index name for the sample.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1,
        help="Number of Elasticsearch results to return.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Search query. You can also pass it as a positional argument after search.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.task == "sentiment":
        run_sentiment_sample(args.data_path, output_dir=args.output_dir)

    if args.task == "index":
        try:
            run_index_sample(index_name=args.index_name)
        except Exception as exc:
            print(f"Elasticsearch index sample skipped: {exc}")

    if args.task == "search":
        try:
            query = args.query_text or args.query
            if not query:
                raise ValueError(
                    'Search query is required. Example: python sample.py search "battery"'
                )
            run_search_sample(
                index_name=args.index_name,
                query=query,
                size=args.size,
            )
        except Exception as exc:
            print(f"Elasticsearch search sample skipped: {exc}")


if __name__ == "__main__":
    main()
