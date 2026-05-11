# Social Intelligence Lab Sample

이 저장소는 리뷰 데이터를 이용한 감성 분석과 Elasticsearch 검색 실습용 코드입니다.
`sample.py`는 전체 흐름을 확인하기 위한 예시 코드이고, 실제 과제는 `sentiment.py`와
`elastic_search.py`의 TODO를 채우는 방식으로 진행합니다.

압축 해제 시, 폴더/파일 구조
```text
Social_Intelligence_Final_Project/
├── .env.example
├── elastic_search.py
├── pyproject.toml
├── README.md
├── sample.py
├── sentiment.py
└── si_dataset/
    ├── review_for_analysis.json
    └── train_review_data.json
```

## 환경 설정

필요한 패키지를 설치합니다.

```bash
uv sync
```

환경 변수 예시는 `.env.example`을 참고합니다.
`.env.example`을 복사해서 `.env` 파일을 만들고, 각자 환경에 맞게 값을 채워서
환경 변수를 관리합니다.

```bash
cp .env.example .env
```

Elasticsearch를 사용하는 실습은 Elasticsearch 서버가 실행 중이어야 합니다.
실행 방법은 운영체제, 설치 방식, Docker 사용 여부에 따라 다를 수 있습니다.

## sample.py 실행

`sample.py`는 랜덤 감성 분석, 더미 데이터 인덱싱, 랜덤 검색 예시를 제공합니다.
기본 감성 분석 데이터는 `si_dataset/train_review_data.json`입니다.

감성 분석만 실행:

```bash
python sample.py sentiment
```

입력 파일과 output 폴더를 지정해서 실행:

```bash
python sample.py sentiment \
  --data-path si_dataset/train_review_data.json \
  --output-dir output \
  --seed 42
```

더미 데이터를 Elasticsearch에 저장:

```bash
python sample.py index --index-name sample_reviews
```

`sample.py index`는 더미 데이터를 먼저 전처리한 뒤, 전처리된 documents를
Elasticsearch에 저장합니다. 실제 과제에서도 원본 데이터를 바로 넣지 말고, 먼저
정리된 documents를 만든 뒤 index에 넣는 흐름을 권장합니다.

Elasticsearch에서 검색어로 검색:

```bash
python sample.py search "배터리" --index-name sample_reviews --size 1
```

인덱싱과 검색을 함께 확인하려면 `index`를 먼저 실행한 뒤 `search`를 실행합니다.
더미 데이터나 mapping을 바꾼 뒤에는 기존 index를 다시 만들어야 하므로 `index`를 다시 실행하세요.

```bash
python sample.py index --index-name sample_reviews
python sample.py search "배터리" --index-name sample_reviews --size 1
```

한국어 검색에서는 analyzer에 따라 `배터리도`와 `배터리`가 서로 매칭되지 않을 수 있습니다.
실제 과제에서는 한국어 형태소 분석기, ngram analyzer, synonym, hybrid search 등을 고려할 수 있습니다.

감성 분석 결과는 `output/` 폴더에 JSON으로 저장됩니다. 파일명에는 날짜와 시간이
포함되어 중복 저장을 피합니다.

저장되는 JSON에는 다음 내용이 들어갑니다.

- `input_file`: 사용한 데이터 파일
- `num_records`: 처리한 리뷰 개수
- `score`: quadratic weighted kappa 점수
- `evaluation`: 평가 지표와 점수
- `predictions`: `review_id`, `pred` 목록

## TODO

아래 항목들은 코드 구현을 위해 필수적인 내용과 직접 구현/수정해야 하는 부분입니다.
`sample.py`는 각 항목의 가장 단순한 예시를 보여주는 파일입니다.

### sentiment.py

`run_sentiment_analysis(...)`

- 감성 분석 평가와 결과 저장을 진행하는 공통 실행 함수입니다.
- 이 함수는 과제 평가 흐름에 사용되므로 수정하지 마세요.
- 변경이 필요하다고 판단되면 먼저 저에게 문의하세요.

`preprocess_content(content: str) -> str`

- 감성 분석 전에 리뷰 텍스트를 전처리합니다.
- 예: `strip()`, HTML 제거, 불필요한 공백 정리, 특수문자 처리, 토큰화 등
- `sample.py`에서는 예시로 `content.strip()`만 적용합니다.

`predict_sentiment(content: str) -> int`

- 전처리된 리뷰 텍스트를 입력받아 감성 Label을 예측합니다.
- 예측 모델은 학습/Pre-trained 모델 활용/Feature engineering 등을 자유롭게 하면 됩니다.
  - 통계 기반 전처리의 경우, training dataset에서 적용한 값을 활용해야 테스트셋에서도 적용이 가능할수 있습니다.
- Label은 반드시 다음 중 하나여야 합니다.
- `0`: negative
- `1`: weak_negative
- `2`: weak_positive
- `3`: positive

### elastic_search.py

`DEFAULT_INDEX_MAPPING`

- Elasticsearch index에 저장할 documents의 필드 타입을 정의합니다.
- `preprocess_record()`가 만들어내는 document 구조와 mapping이 서로 맞아야 합니다.
- 예: `review_id`는 `keyword`, 본문 검색용 `content`는 `text`, 숫자 필드는 `integer` 등

`preprocess_record(record: dict) -> dict`

- Elasticsearch에 넣기 전에 리뷰 레코드를 문서 형태로 전처리합니다.
- 예: 필드 정규화, HTML 제거, 빈 텍스트 제거, 길이가 너무 짧은 리뷰 제거, 메타데이터 추가 등
- 이 함수는 원본 데이터에서 index에 넣을 documents를 만드는 단계입니다.
- 전처리 결과 document에는 최소한 `review_id`, `content`가 포함되어야 합니다.

`build_documents(records: list[dict]) -> list[dict]`

- 여러 개의 원본 record에 `preprocess_record()`를 적용해 documents를 만듭니다.
- 이 함수가 자동으로 호출되는 것은 아닙니다.
- 실제 흐름은 `documents = build_documents(records)`로 먼저 documents를 만든 뒤,
  `index_documents(documents=documents, ...)`에 넣는 방식입니다.

`build_search_body(query: str, size: int) -> dict`

- 검색 요청에 사용할 Elasticsearch query body를 만듭니다.
- 예: `match`, `multi_match`, 필터, query expansion, hybrid search 등
- `sample.py`에서는 `content`와 `product_name`에 대한 간단한 `multi_match` 예시를 보여줍니다.

`search(query: str, ...) -> SearchOutput`

- 전체 retrieval pipeline을 구현하는 함수입니다.
- `build_search_body()`로 query body를 만들고, Elasticsearch에 검색 요청을 보내고,
  raw hit를 정규화한 뒤, `postprocess_results()`까지 적용해 최종 `SearchOutput`을 반환합니다.
- lexical search, hybrid search, vector search, reranking 등은 이 함수 안에서 자유롭게 조합할 수 있습니다.
- 최종 반환값은 `validate_search_output()`을 통과해야 합니다.

`postprocess_results(results: list[dict], query: str) -> SearchOutput`

- 검색 결과가 나온 뒤 후처리하는 단계입니다.
- 예: reranking, 중복 제거, 점수 기준 필터링, LLM 기반 relevance 판단, 결과 포맷 변경 등
- 최종 출력은 반드시 `SearchOutput` 형태를 지켜야 합니다.

```python
{
    "query": query,
    "results": [
        {
            "review_id": "...",
            "score": 1.23,
            "product_name": "...",
            "content": "...",
        }
    ],
}
```

- `query`: 사용자가 입력한 검색어입니다.
- `results`: 검색 결과 목록입니다.
- `review_id`: Elasticsearch의 `_id`를 `id`로 받은 값이나 document의 `review_id`를 사용하면 됩니다.
- `score`: Elasticsearch score 또는 reranking 이후 점수입니다.
- `product_name`: 제품명입니다. 없으면 `None`을 사용할 수 있습니다.
- `content`: 검색 결과로 보여줄 리뷰 본문입니다.
- `sample.py`에서는 reranking 없이 이 schema로만 정리해서 반환합니다.

`validate_search_output(output) -> bool`

- `postprocess_results()`의 출력이 위 schema를 지키는지 검사합니다.
- 최종 제출 전 이 검사를 통과하는지 확인하세요.

필수 schema는 코드에서는 다음 타입으로 정의되어 있습니다.

```python
class SearchOutputItem(TypedDict):
    review_id: str
    score: float
    product_name: str | None
    content: str


class SearchOutput(TypedDict):
    query: str
    results: list[SearchOutputItem]
```

sample 검색 흐름은 다음과 같습니다.

```python
output = search(query=query, index_name=index_name, size=size)
assert elastic_search.validate_search_output(output)
```

## Analysis 데이터 주의사항

`si_dataset/review_for_analysis.json` 같은 analysis 데이터는 검색 index에 넣기 전에
반드시 전처리해야 합니다. 별도 함수가 없더라도, Elasticsearch에 넣기 전 단계에서
품질이 낮은 데이터를 걸러내는 과정이 필요합니다.

특히 다음과 같은 데이터는 그대로 넣지 않는 것이 좋습니다.

- 내용이 비어 있거나 너무 짧은 리뷰
- HTML, 스크립트, 깨진 문자 등이 많이 섞인 리뷰
- 중복 리뷰
- 제품명, 리뷰 ID, 본문 등 필수 필드가 없는 레코드
- 검색 결과 품질을 떨어뜨리는 광고성/무의미한 텍스트

검색 성능은 모델이나 query body만으로 결정되지 않습니다. index에 들어가는 데이터의
품질이 검색 결과 품질을 크게 좌우하므로, analysis 데이터는 먼저 정리하고 선별한 뒤
Elasticsearch에 저장해야 합니다.
