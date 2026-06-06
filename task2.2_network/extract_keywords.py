from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "task2.2_network" / "data" / "cleaned_reviews.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "task2.2_network" / "data"

TEXT_FIELD_NAMES = ("title", "content")

STOPWORDS = {
    "제목",
    "본문",
    "설문",
    "그리고",
    "하지만",
    "그래서",
    "정말",
    "너무",
    "매우",
    "완전",
    "진짜",
    "그냥",
    "약간",
    "일단",
    "또",
    "더",
    "좀",
    "거",
    "것",
    "수",
    "등",
    "때",
    "중",
    "제",
    "저",
    "나",
    "이번",
    "이제",
    "사용중",
    "합니다",
    "있습니다",
    "없습니다",
    "같아요",
    "같습니다",
    "됩니다",
    "했어요",
    "네요",
    "이네요",
    "이에요",
    "예요",
    "쿠팡",
    "아이폰",
    "갤럭시",
    "애플",
    "삼성",
    "제품",
    "생각",
    "정도",
    "부분",
    "느낌",
    "사람",
    "사용",
    "모델",
    "시리즈",
    "후기",
    "마음",
    "처음",
    "기존",
    "프로",
    "맥스",
    "울트라",
    "플립",
    "폴드",
    "s",
    "and",
    "the",
    "with",
    "for",
    "to",
    "i",
    "it",
    "pro",
    "max",
    "galaxy",
    "samsung",
    "iphone",
    "apple",
    "c",
    "g",
    "리뷰",
    "상품",
    "전자",
    "좋다",
    "있다",
    "쓰다",
    "하다",
    "되다",
    "없다",
    "같다",
    "크다",
    "들다",
    "사다",
    "느끼다",
    "나오다",
    "바꾸다",
    "보이다",
    "오다",
    "가다",
    "높다",
    "필요",
    "전체",
}

CANONICAL_KEYWORDS = {
    "디스플레": "디스플레이",
    "자급": "자급제",
    "밧데리": "배터리",
    "베터리": "배터리",
    "뾱뾱이": "뽁뽁이",
    "뽁뽁": "뽁뽁이",
    "뾱뾱": "뽁뽁이",
}

NOUN_KEYWORDS = {
    "가성비",
    "가격",
    "감성",
    "개통",
    "결함",
    "고장",
    "교환",
    "구매",
    "기기",
    "내구성",
    "디스플레이",
    "디자인",
    "렌즈",
    "무게",
    "발열",
    "반응속도",
    "반품",
    "밝기",
    "방전",
    "배송",
    "배터리",
    "버벅임",
    "보험",
    "불량",
    "사은품",
    "사진",
    "색감",
    "색상",
    "성능",
    "셀룰러",
    "속도",
    "스크래치",
    "스피커",
    "실물",
    "액정",
    "야간",
    "용량",
    "유심",
    "음질",
    "자급제",
    "주사율",
    "충전",
    "카메라",
    "케이스",
    "터치",
    "통화",
    "포장",
    "품질",
    "할인",
    "해상도",
    "화면",
    "화질",
    "환불",
}

PHRASE_KEYWORDS: dict[str, tuple[str, list[str]]] = {
    "가격 부담": ("phrase", ["가격 부담", "부담스러운 가격", "부담스럽"]),
    "가성비 좋다": ("phrase", ["가성비 좋", "가격 좋", "저렴하게", "싸게"]),
    "가성비 나쁘다": ("phrase", ["가성비 별로", "성능에 비해 비싸", "가격 비싸", "비싸요"]),
    "고속 충전": ("phrase", ["고속충전", "고속 충전", "충전 빠르"]),
    "구매 만족": ("phrase", ["구매 만족", "사길 잘", "잘 샀", "잘산", "잘 샀다"]),
    "기기 결함": ("phrase", ["기기 결함", "제품 결함", "기계 문제"]),
    "내구성 약하다": ("phrase", ["내구성 약", "약한 내구성", "찌그러져", "찍힘"]),
    "디자인 만족": ("phrase", ["디자인 만족", "디자인 예쁘", "실물 예쁘", "색상 예쁘"]),
    "디자인 불만": ("phrase", ["디자인 별로", "디자인 아쉽", "마음에 안 들"]),
    "렌즈 전환": ("phrase", ["렌즈 전환", "카메라 렌즈"]),
    "무게 가볍다": ("phrase", ["가벼워요", "가볍게", "가볍습니다", "무게 가볍"]),
    "무게 무겁다": ("phrase", ["무거워요", "무겁습니다", "너무 무겁", "무게 무겁"]),
    "배송 빠르다": ("phrase", ["배송 빠르", "빨리 도착", "빠르게 도착", "새벽 배송"]),
    "배송 불만": ("phrase", ["배송 불만", "배송 엉망", "배송 문제", "배송 업체"]),
    "배터리 오래감": ("phrase", ["배터리 오래", "오래가", "하루종일", "하루 종일", "사용 시간 길"]),
    "배터리 짧음": ("phrase", ["배터리 짧", "빨리 닳", "금방 닳", "생각보다짧"]),
    "발열 적다": ("phrase", ["발열 없", "발열 적", "뜨겁지 않"]),
    "발열 심하다": ("phrase", ["발열 심", "뜨거워", "뜨겁", "열감 심"]),
    "반품 고민": ("phrase", ["반품 고민", "반품할까", "환불하거나 교환", "교환할 생각"]),
    "불량 의심": ("phrase", ["불량", "뽑기", "고장인가", "문제 있"]),
    "뽁뽁이 없음": ("phrase", ["뽁뽁이 없이", "뾱뾱이 없이", "완충재 없이", "안감싸진"]),
    "성능 좋다": ("phrase", ["성능 좋", "속도 빠르", "빠릿", "쾌적"]),
    "성능 아쉽다": ("phrase", ["성능 아쉽", "버벅", "렉", "느려"]),
    "스피커 불만": ("phrase", ["스피커 안 좋", "스피커 별로", "음질 별로"]),
    "완충 포장": ("phrase", ["완충 포장", "꼼꼼하게 포장", "포장 꼼꼼", "뽁뽁이", "뾱뾱이"]),
    "카메라 만족": ("phrase", ["카메라 만족", "화질 좋", "사진 잘", "선명한 화질"]),
    "카메라 불만": ("phrase", ["카메라 별로", "화질 별로", "사진 흐", "카메라 실망"]),
    "화면 선명": ("phrase", ["화면 선명", "디스플레이 선명", "화질 선명", "밝고 선명"]),
}

STEM_KEYWORDS: dict[str, tuple[str, str]] = {
    "가볍": ("가볍다", "adjective"),
    "고민": ("고민하다", "verb"),
    "구매": ("구매하다", "verb"),
    "괜찮": ("괜찮다", "adjective"),
    "나쁘": ("나쁘다", "adjective"),
    "느리": ("느리다", "adjective"),
    "느려": ("느리다", "adjective"),
    "닳": ("닳다", "verb"),
    "도착": ("도착하다", "verb"),
    "만족": ("만족하다", "verb"),
    "무겁": ("무겁다", "adjective"),
    "반품": ("반품하다", "verb"),
    "별로": ("별로", "adjective"),
    "부드럽": ("부드럽다", "adjective"),
    "빠르": ("빠르다", "adjective"),
    "빠릿": ("빠릿하다", "adjective"),
    "비싸": ("비싸다", "adjective"),
    "선명": ("선명하다", "adjective"),
    "실망": ("실망하다", "verb"),
    "아쉽": ("아쉽다", "adjective"),
    "예쁘": ("예쁘다", "adjective"),
    "오래": ("오래가다", "verb"),
    "추천": ("추천하다", "verb"),
    "충전": ("충전하다", "verb"),
    "편하": ("편하다", "adjective"),
    "환불": ("환불하다", "verb"),
    "후회": ("후회하다", "verb"),
}


class MorphAnalyzer(Protocol):
    name: str

    def extract(self, text: str) -> list[tuple[str, str]]:
        """Return normalized keyword candidates as (text, pos_group)."""


class KiwiAnalyzer:
    name = "kiwi"

    def __init__(self) -> None:
        from kiwipiepy import Kiwi

        self._kiwi = Kiwi()

    def extract(self, text: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for token in self._kiwi.tokenize(text):
            form = normalize_morph_form(token.lemma)
            tag = token.tag
            if not form:
                continue
            if tag.startswith("NN") or tag == "SL":
                candidates.append((form, "noun"))
            elif tag == "VA":
                candidates.append((canonical_predicate(form), "adjective"))
            elif tag == "VV":
                candidates.append((canonical_predicate(form), "verb"))
        return candidates


class OktAnalyzer:
    name = "okt"

    def __init__(self) -> None:
        from konlpy.tag import Okt

        self._okt = Okt()

    def extract(self, text: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for form, tag in self._okt.pos(text, stem=True):
            form = normalize_morph_form(form)
            if not form:
                continue
            if tag == "Noun":
                candidates.append((form, "noun"))
            elif tag == "Adjective":
                candidates.append((canonical_predicate(form), "adjective"))
            elif tag == "Verb":
                candidates.append((canonical_predicate(form), "verb"))
        return candidates


class MecabAnalyzer:
    name = "mecab"

    def __init__(self) -> None:
        try:
            from konlpy.tag import Mecab

            self._mecab = Mecab()
        except Exception:
            import MeCab

            self._tagger = MeCab.Tagger()
            self._mecab = None

    def extract(self, text: str) -> list[tuple[str, str]]:
        if self._mecab is not None:
            return self._extract_konlpy(text)
        return self._extract_native(text)

    def _extract_konlpy(self, text: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for form, tag in self._mecab.pos(text):
            form = normalize_morph_form(form)
            if not form:
                continue
            if tag.startswith("NN") or tag == "SL":
                candidates.append((form, "noun"))
            elif tag == "VA":
                candidates.append((canonical_predicate(form), "adjective"))
            elif tag == "VV":
                candidates.append((canonical_predicate(form), "verb"))
        return candidates

    def _extract_native(self, text: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        parsed = self._tagger.parse(text)
        for line in parsed.splitlines():
            if line == "EOS" or "\t" not in line:
                continue
            form, features = line.split("\t", 1)
            tag = features.split(",", 1)[0]
            form = normalize_morph_form(form)
            if not form:
                continue
            if tag.startswith("NN") or tag == "SL":
                candidates.append((form, "noun"))
            elif tag == "VA":
                candidates.append((canonical_predicate(form), "adjective"))
            elif tag == "VV":
                candidates.append((canonical_predicate(form), "verb"))
        return candidates


class RegexFallbackAnalyzer:
    name = "regex_fallback"

    def extract(self, text: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for token in tokenize(text):
            normalized = normalize_token(token)
            if not normalized:
                continue
            if normalized in NOUN_KEYWORDS:
                candidates.append((normalized, "noun"))
                continue
            for stem, (canonical, pos_group) in STEM_KEYWORDS.items():
                if normalized.startswith(stem):
                    candidates.append((canonical, pos_group))
                    break
            else:
                if len(normalized) >= 3 and normalized.endswith(("감", "성", "력", "율")):
                    candidates.append((normalized, "noun"))
        return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract review keywords and Review-Keyword edges."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Drop keywords that appear in fewer than this many reviews.",
    )
    parser.add_argument(
        "--max-keywords-per-review",
        type=int,
        default=40,
        help="Keep the top N keywords per review after global df filtering.",
    )
    parser.add_argument(
        "--analyzer",
        choices=("auto", "kiwi", "okt", "mecab", "regex"),
        default="auto",
        help="Morph analyzer. auto tries Kiwi, Okt, Mecab, then regex fallback.",
    )
    return parser.parse_args()


def keyword_id(normalized: str) -> str:
    safe = sanitize_keyword_text(normalized).replace(" ", "_")
    safe = re.sub(r"[^0-9a-zA-Z가-힣_]+", "", safe)
    return f"kw:{safe}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", text)


def normalize_morph_form(form: str) -> str:
    form = normalize_token(form)
    form = sanitize_keyword_text(form)
    if not form or form in STOPWORDS:
        return ""
    if form.isdigit():
        return ""
    if len(form) < 2:
        return ""
    return form


def canonical_predicate(form: str) -> str:
    if not form:
        return ""
    if form.endswith("다"):
        return form
    return f"{form}다"


def normalize_token(token: str) -> str:
    token = unicodedata.normalize("NFKC", token).strip().lower()
    token = re.sub(r"(이에요|예요|입니다|합니다|했어요|해요|네요|군요|어요|아요|으로|에서|에게|보다|까지|부터|처럼|만큼|이라도|라도|이나|나|은|는|이|가|을|를|도|만|의)$", "", token)
    return token.strip()


def sanitize_keyword_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣\s_]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[0-9a-zA-Z가-힣]", text):
        return ""
    return text


def add_keyword(
    bucket: dict[str, dict[str, Any]],
    normalized: str,
    pos_group: str,
    source: str,
    count: int = 1,
) -> None:
    normalized = sanitize_keyword_text(normalized)
    normalized = CANONICAL_KEYWORDS.get(normalized, normalized)
    if not normalized or normalized in STOPWORDS:
        return
    if normalized.isdigit():
        return
    if len(normalized) < 2:
        return

    item = bucket.setdefault(
        normalized,
        {
            "keyword_id": keyword_id(normalized),
            "text": normalized,
            "normalized": normalized,
            "pos_group": pos_group,
            "count": 0,
            "sources": set(),
        },
    )
    item["count"] += count
    item["sources"].add(source)
    if item["pos_group"] == "unknown" and pos_group != "unknown":
        item["pos_group"] = pos_group
    if item["pos_group"] == "noun" and pos_group == "phrase":
        item["pos_group"] = pos_group


def select_analyzer(name: str) -> MorphAnalyzer:
    analyzer_classes: list[type[MorphAnalyzer]]
    if name == "auto":
        analyzer_classes = [KiwiAnalyzer, OktAnalyzer, MecabAnalyzer, RegexFallbackAnalyzer]
    elif name == "kiwi":
        analyzer_classes = [KiwiAnalyzer]
    elif name == "okt":
        analyzer_classes = [OktAnalyzer]
    elif name == "mecab":
        analyzer_classes = [MecabAnalyzer]
    else:
        analyzer_classes = [RegexFallbackAnalyzer]

    errors = []
    for analyzer_class in analyzer_classes:
        try:
            return analyzer_class()
        except Exception as exc:
            errors.append(f"{analyzer_class.name}: {exc!r}")
            continue
    raise RuntimeError("No analyzer could be initialized: " + " / ".join(errors))


def extract_from_field(
    text: str,
    source: str,
    bucket: dict[str, dict[str, Any]],
    analyzer: MorphAnalyzer,
) -> None:
    lowered = text.lower()

    for normalized, (pos_group, variants) in PHRASE_KEYWORDS.items():
        count = 0
        for variant in variants:
            count += len(re.findall(re.escape(variant.lower()), lowered))
        if count:
            add_keyword(bucket, normalized, pos_group, source, count)

    for normalized, pos_group in analyzer.extract(text):
        add_keyword(bucket, normalized, pos_group, source)


def extract_from_survey_answers(
    survey_answers: list[dict[str, str]],
    bucket: dict[str, dict[str, Any]],
) -> None:
    for survey in survey_answers:
        question = str(survey.get("question") or "")
        answer = str(survey.get("answer") or "")
        combined = f"{question} {answer}"

        if question == "가성비":
            if any(term in answer for term in ("비싸", "별로", "부담")):
                add_keyword(bucket, "가성비 나쁘다", "phrase", "survey_text")
            elif any(term in answer for term in ("좋", "저렴", "적당")):
                add_keyword(bucket, "가성비 좋다", "phrase", "survey_text")
        elif question == "사용 시간":
            if any(term in answer for term in ("짧", "빨리", "부족")):
                add_keyword(bucket, "배터리 짧음", "phrase", "survey_text")
            elif any(term in answer for term in ("길", "오래", "충분")):
                add_keyword(bucket, "배터리 오래감", "phrase", "survey_text")
        elif question == "디자인":
            if any(term in answer for term in ("별로", "안", "아쉽")):
                add_keyword(bucket, "디자인 불만", "phrase", "survey_text")
            elif any(term in answer for term in ("만족", "예", "좋")):
                add_keyword(bucket, "디자인 만족", "phrase", "survey_text")
        elif question == "카메라 성능":
            if any(term in answer for term in ("별로", "안", "아쉽")):
                add_keyword(bucket, "카메라 불만", "phrase", "survey_text")
            elif any(term in answer for term in ("뛰어나", "만족", "좋")):
                add_keyword(bucket, "카메라 만족", "phrase", "survey_text")
        elif question == "무게":
            if "무거" in answer:
                add_keyword(bucket, "무게 무겁다", "phrase", "survey_text")
            elif "가벼" in answer:
                add_keyword(bucket, "무게 가볍다", "phrase", "survey_text")

def extract_review_keywords(
    document: dict[str, Any],
    analyzer: MorphAnalyzer,
) -> list[dict[str, Any]]:
    bucket: dict[str, dict[str, Any]] = {}
    for source in TEXT_FIELD_NAMES:
        value = str(document.get(source) or "")
        if value:
            extract_from_field(value, source, bucket, analyzer)
    extract_from_survey_answers(document.get("survey_answers") or [], bucket)

    keywords = []
    for item in bucket.values():
        item["sources"] = sorted(item["sources"])
        keywords.append(item)
    keywords.sort(key=lambda item: (-item["count"], item["normalized"]))
    return keywords


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, documents: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviews = load_jsonl(args.input)
    analyzer = select_analyzer(args.analyzer)

    review_keywords_raw: list[dict[str, Any]] = []
    keyword_df: Counter[str] = Counter()
    keyword_tf: Counter[str] = Counter()
    keyword_pos: dict[str, str] = {}

    for review in reviews:
        keywords = extract_review_keywords(review, analyzer)
        review_keywords_raw.append({"review": review, "keywords": keywords})
        for keyword in keywords:
            normalized = keyword["normalized"]
            keyword_df[normalized] += 1
            keyword_tf[normalized] += int(keyword["count"])
            keyword_pos.setdefault(normalized, keyword["pos_group"])

    allowed_keywords = {
        normalized for normalized, df in keyword_df.items() if df >= args.min_df
    }

    review_keyword_docs: list[dict[str, Any]] = []
    edge_docs: list[dict[str, Any]] = []
    filtered_review_keyword_sets: dict[str, list[dict[str, Any]]] = {}

    for item in review_keywords_raw:
        review = item["review"]
        filtered = [
            keyword
            for keyword in item["keywords"]
            if keyword["normalized"] in allowed_keywords
        ][: args.max_keywords_per_review]
        filtered_review_keyword_sets[review["review_id"]] = filtered

        review_keyword_docs.append(
            {
                "review_id": review["review_id"],
                "product_name": review["product_name"],
                "rating": review["rating"],
                "sentiment_id": review["sentiment_id"],
                "keywords": filtered,
            }
        )
        for keyword in filtered:
            edge_docs.append(
                {
                    "review_id": review["review_id"],
                    "product_name": review["product_name"],
                    "sentiment_id": review["sentiment_id"],
                    "keyword_id": keyword["keyword_id"],
                    "keyword_text": keyword["text"],
                    "keyword_normalized": keyword["normalized"],
                    "pos_group": keyword["pos_group"],
                    "count": keyword["count"],
                    "sources": keyword["sources"],
                    "weight": 1.0,
                }
            )

    keyword_docs = []
    product_df: dict[str, Counter[str]] = defaultdict(Counter)
    sentiment_df: dict[str, Counter[str]] = defaultdict(Counter)
    for item in review_keywords_raw:
        review = item["review"]
        seen = {
            keyword["normalized"]
            for keyword in item["keywords"]
            if keyword["normalized"] in allowed_keywords
        }
        for normalized in seen:
            product_df[normalized][review["product_name"]] += 1
            sentiment_df[normalized][review["sentiment_id"]] += 1

    for normalized in sorted(allowed_keywords):
        keyword_docs.append(
            {
                "keyword_id": keyword_id(normalized),
                "text": normalized,
                "normalized": normalized,
                "pos_group": keyword_pos.get(normalized, "unknown"),
                "df": keyword_df[normalized],
                "tf": keyword_tf[normalized],
                "product_df": dict(sorted(product_df[normalized].items())),
                "sentiment_df": dict(sorted(sentiment_df[normalized].items())),
                "is_stopword": False,
            }
        )

    output_keywords = args.output_dir / "keywords.jsonl"
    output_review_keywords = args.output_dir / "review_keywords.jsonl"
    output_edges = args.output_dir / "review_keyword_edges.jsonl"
    output_summary = args.output_dir / "keyword_extraction_summary.json"

    write_jsonl(output_keywords, keyword_docs)
    write_jsonl(output_review_keywords, review_keyword_docs)
    write_jsonl(output_edges, edge_docs)

    empty_reviews = sum(1 for doc in review_keyword_docs if not doc["keywords"])
    keywords_per_review = [len(doc["keywords"]) for doc in review_keyword_docs]
    pos_counts = Counter(doc["pos_group"] for doc in keyword_docs)
    top_keywords = sorted(
        keyword_docs,
        key=lambda item: (-item["df"], -item["tf"], item["normalized"]),
    )[:50]

    summary = {
        "input_path": str(args.input),
        "output_keywords": str(output_keywords),
        "output_review_keywords": str(output_review_keywords),
        "output_edges": str(output_edges),
        "review_count": len(reviews),
        "keyword_count": len(keyword_docs),
        "edge_count": len(edge_docs),
        "min_df": args.min_df,
        "max_keywords_per_review": args.max_keywords_per_review,
        "analyzer": analyzer.name,
        "empty_review_count": empty_reviews,
        "keywords_per_review": {
            "min": min(keywords_per_review, default=0),
            "max": max(keywords_per_review, default=0),
            "avg": round(sum(keywords_per_review) / len(keywords_per_review), 2)
            if keywords_per_review
            else 0,
        },
        "pos_group_counts": dict(sorted(pos_counts.items())),
        "top_keywords": [
            {
                "keyword": item["normalized"],
                "pos_group": item["pos_group"],
                "df": item["df"],
                "tf": item["tf"],
            }
            for item in top_keywords
        ],
    }

    with output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
