import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from elastic_search import build_documents


def parse_args():
    parser = argparse.ArgumentParser(description="Preview preprocessed retrieval documents.")
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of raw records to preview.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable OpenAI API-based metadata enrichment.",
    )
    # 추가
    #parser.add_argument(
    #    "--save",
    #    type=str,
    #    default=None,
    #    help="Path to save the processed documents as a JSON file.",
    #)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = ROOT_DIR / "si_dataset" / "review_for_analysis.json"

    with data_path.open(encoding="utf-8") as f:
        records = json.load(f)[: args.limit]

    docs = build_documents(records, enrich_with_llm=args.llm)

    print(f"raw records: {len(records)}")
    print(f"documents: {len(docs)}")
    print(f"llm enrichment: {args.llm}")

    # 추가
    #if args.save:
    #    with open(args.save, "w", encoding="utf-8") as f:
    #        json.dump(docs, f, ensure_ascii=False, indent=2)
    #    print(f"\n✅ 프로세싱 완료된 문서를 다음 경로에 저장했습니다: {args.save}")

    for i, doc in enumerate(docs):
        print(f"\n=== Document {i+1} ===")
        print(json.dumps(doc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
