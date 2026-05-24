# Task 2 Retrieval Design

이 문서는 `elastic_search.py`에 hybrid retrieval을 구현하기 전, 검색 파이프라인과 평가 기준을 고정하기 위한 설계 문서입니다.

## 목표

스마트폰 e-commerce 리뷰 데이터에서 사용자의 자연어 질의에 맞는 리뷰를 top 10으로 반환한다.

최종 출력은 `elastic_search.SearchOutput` 형식을 따른다.

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

## 데이터와 평가셋

debugging용 자체 평가셋은 `task2_retrieval/eval/human_eval.json`에 둔다.

현재 평가셋은 15개 query로 구성되어 있으며, 각 query마다 사람이 고른 관련 document top 10이 들어 있다.

평가셋의 query category:

- `1_Short_Keyword`: 짧은 키워드 중심 질의
- `2_Medium_Context`: 맥락이 있는 중간 길이 질의
- `3_Long_Complex`: 조건이 여러 개인 긴 복합 질의

각 query entry에는 다음 정보가 포함된다.

- `query`: 검색 질의 원문
- `category`: query 난이도/형태
- `intent`: 사람이 정의한 검색 의도
- `per_model.openai`: 사람이 검수한 관련 review top 10
- `reviewId`, `rank`, `score`, `sentiment`, `evidence`, `reasoning`, `content`

초기 디버깅에서는 다음을 우선 확인한다.

- top 10 안에 human_eval의 정답 reviewId가 얼마나 포함되는지
- query intent의 `must_match` 조건을 만족하는 문서가 상위에 오는지
- `exclude` 조건에 걸리는 문서가 상위에 섞이지 않는지
- 짧은 키워드 query와 긴 복합 query 모두에서 recall이 유지되는지

### Evaluation Metric: MAP

공식 평가지표는 MAP(Mean Average Precision)로 둔다. MAP는 관련 문서가 top rank에 얼마나 빨리 등장하는지를 평가하므로, 단순히 top 10 안에 포함시키는 것보다 상위 순서가 중요하다.

query별 AP(Average Precision):

```text
AP(q) = sum(P@k * rel(k)) / number_of_relevant_docs
```

- `rel(k)`: rank k의 문서가 relevant이면 1, 아니면 0
- `P@k`: rank k까지의 precision
- `number_of_relevant_docs`: 해당 query의 gold relevant document 수

전체 MAP:

```text
MAP = mean(AP(q) for q in queries)
```

`human_eval.json` 기준 디버깅에서는 `per_model.openai`의 reviewId 목록을 relevant set으로 사용한다. 현재 각 query에 top 10 gold가 있으므로, 내부 평가에서는 `AP@10`과 `MAP@10`을 우선 본다.

MAP 최적화 관점에서 중요한 점:

- gold 문서가 top 10에 들어오는 것만으로는 부족하고, rank 1-3에 많이 올라와야 한다.
- RRF 단계는 recall 확보가 목적이고, MAP 개선은 주로 reranking과 post filtering에서 발생한다.
- post filtering은 관련 문서를 제거하면 MAP가 크게 떨어질 수 있으므로 hard filter보다 penalty 기반 정렬 보정을 우선한다.

## Index Fields

현재 `DEFAULT_INDEX_MAPPING` 기준 주요 필드는 다음과 같다.

### 원본/표시 필드

- `review_id`: Elasticsearch `_id`로도 사용하는 고유 ID
- `review_at`: 리뷰 작성 시점
- `product_name`: 상품 그룹명
- `item_name`: 옵션/상세 상품명
- `title`: 리뷰 제목
- `content`: 리뷰 본문
- `rating`: 별점
- `helpful_count`, `helpful_true_count`, `helpful_false_count`: 도움돼요 관련 수치
- `has_image`, `has_video`: 첨부 여부

### LLM enrichment 필드

- `issue_category`: 배송/포장, 배터리/충전, 발열 등 고정 issue category 상위 3개
- `aspect_keywords`: 리뷰에서 중요하게 걸리는 자유형 aspect keyword phrase 5개
- `issue_sentiments`: 상위 1-3개 issue category에 대한 sentiment 구조화 결과
  - `issue_category`: 고정 issue category label
  - `sentiment`: `positive`, `negative`, `mixed`, `neutral`
  - `evidence`: 리뷰에 근거한 짧은 판단 근거
- `short_summary`: 80자 이내 요약

`aspect_keywords`와 `issue_category`는 목적이 다르다. `aspect_keywords`는 "뽁뽁이", "완충 포장", "배터리 오래감"처럼 자유롭게 뽑는 세부 표현이고, `issue_category`는 retrieval에서 비교하기 쉬운 고정 label이다. `issue_sentiments`는 이 고정 issue category별 sentiment를 저장한다. 예를 들어 "제품은 만족하지만 뽁뽁이 포장은 아쉬움"은 전체 sentiment만 보면 positive일 수 있지만, `배송/포장` issue에서는 negative로 저장되어야 한다.

### Dense retrieval 필드

- `dense_text`: dense embedding 입력용 structured text
- `dense_vector`: `dense_text`를 embedding한 768차원 dense vector

`dense_text`는 단순 concat이 아니라 다음처럼 라벨을 붙인 structured text로 만든다.

```text
상품명: ...
옵션명: ...
제목: ...
본문: ...
요약: ...
이슈 카테고리: 배송/포장, 가격/가성비, 기타
관점 키워드: ...
이슈별 감성: 배송/포장 | positive | 뽁뽁이로 안전하게 포장됨
설문 응답: ...
```

## Hybrid Retrieval Pipeline

최종 검색은 다음 순서로 진행한다.

1. Query expansion
2. BM25 lexical retrieval top 100
3. Dense vector retrieval top 100
4. RRF로 두 candidate list 병합 및 재정렬
5. RRF candidate top 50 선택
6. Cross-encoder reranking
7. Post filtering
8. 최종 top 10 반환

## Step 0. Query Expansion

Query expansion은 retrieval recall을 높이기 위한 단계다. 단, expansion이 원래 의도를 흐리면 MAP가 떨어질 수 있으므로 query type별로 다르게 적용한다. 현재 기본 구현은 LLM이 아니라 rule-based aspect-sentiment expansion이다.

### 기본 원칙

- 원본 query는 항상 가장 높은 weight로 유지한다.
- expansion query는 BM25와 dense에 다르게 사용한다.
- query를 `target_aspect`, `target_sentiment`, `aspect_terms`, `sentiment_terms`, `positive/negative_phrases`로 구조화한다.
- query의 target issue category는 LLM enrichment의 `issue_category`와 같은 고정 label set을 사용한다.
- 짧은 keyword query에는 synonym/variant 확장을 강하게 쓰되 LLM rewrite는 보수적으로 쓴다.
- 긴 complex query에는 의미 보존 rewrite와 intent extraction을 쓴다.
- negation, sentiment, aspect 조건은 expansion 중 절대 제거하지 않는다.

### Query Type별 전략

`1_Short_Keyword`:

- 예: "뽁뽁이 포장 상태에 만족한 리뷰"
- 목표: 표현 변형과 동의어를 넓히되, `배송/포장` aspect에 대한 positive evidence를 우선한다.
- 적용: synonym expansion, typo/variant expansion, aspect keyword expansion, aspect-sentiment phrase boost
- 주의: "만족" query를 "불만" query로 뒤집는 rewrite 금지

`2_Medium_Context`:

- 목표: 핵심 aspect, sentiment, 제품/상황 조건을 분리한다.
- 적용: rule-based aspect/sentiment parsing + structured expansion + 1-2개 paraphrase
- 주의: 너무 많은 paraphrase는 BM25 noise를 늘릴 수 있으므로 제한

`3_Long_Complex`:

- 목표: 긴 query의 조건을 보존하면서 dense retrieval이 이해하기 쉬운 형태로 재작성한다.
- 적용: rule-based parsing으로 가능한 조건을 먼저 잡고, 필요할 때 intent-preserving rewrite, must/include/exclude condition extraction
- 주의: BM25에는 원문 조건을 유지하고, dense에는 자연스러운 rewrite를 추가로 사용

### Rule-Based Aspect-Sentiment Parser

현재 구현은 query를 다음 구조로 변환한다.

```json
{
  "original_query": "...",
  "target_aspect": "배송/포장",
  "target_sentiment": "positive",
  "aspect_terms": ["배송", "포장", "뽁뽁이", "완충", "..."],
  "positive_terms": ["만족", "좋", "꼼꼼", "안전", "..."],
  "negative_terms": ["없이", "파손", "찌그러", "불만", "..."],
  "positive_phrases": ["꼼꼼하게 포장", "안전하게 도착", "..."],
  "negative_phrases": ["뽁뽁이 없이", "박스 파손", "..."]
}
```

중요한 점은 keyword를 단순히 추가하는 것이 아니라, query가 요구하는 sentiment가 어느 aspect에 붙어야 하는지 최대한 보존하는 것이다.

예:

```text
뽁뽁이 포장 상태에 만족한 리뷰
```

는 다음처럼 해석한다.

```text
target_aspect = 배송/포장
target_sentiment = positive
positive evidence = 뽁뽁이/완충/포장이 꼼꼼하거나 안전하게 도착했다는 표현
negative evidence = 뽁뽁이 없이, 박스 파손, 밀봉 뜯김 등
```

### 프로젝트 맞춤 Expansion Dictionary

현재 데이터가 스마트폰 리뷰이므로 다음 사전형 확장을 우선 둔다.

배송/포장:

- `뽁뽁이`: `뾱뾱이`, `에어캡`, `완충재`, `완충 포장`, `보호 포장`
- `포장`: `패키징`, `박스`, `상자`, `택배 포장`
- `파손`: `찌그러짐`, `찍힘`, `구겨짐`, `뜯김`, `흠집`
- `배송`: `택배`, `로켓배송`, `새벽배송`, `도착`

배터리/충전:

- `배터리`: `사용 시간`, `방전`, `오래감`, `하루 종일`, `전작보다 오래`
- `충전`: `충전속도`, `고속충전`, `완충`, `C타입`

카메라:

- `카메라`: `사진`, `셀카`, `화질`, `전면 카메라`, `후면 카메라`, `야간 촬영`
- `선명`: `깨끗`, `디테일`, `밝게`, `잘 나와`

성능/발열:

- `성능`: `속도`, `빠릿`, `버벅`, `렉`, `스크롤`, `부드러움`
- `발열`: `뜨거움`, `미지근`, `열감`, `발열관리`

디자인/무게/화면:

- `디자인`: `색상`, `실물`, `외관`, `고급`, `예쁨`
- `무게`: `가벼움`, `무거움`, `그립감`, `손목`
- `화면`: `디스플레이`, `주사율`, `120Hz`, `밝기`, `AOD`

가격/가성비/AS:

- `가성비`: `가격`, `저렴`, `비싸`, `혜택`, `할인`
- `교환/반품/AS`: `환불`, `반품`, `교환`, `센터`, `보증`

긍정/부정 표현:

- 긍정: `만족`, `좋아요`, `추천`, `잘 왔`, `문제없`, `꼼꼼`, `안전하게`
- 부정: `불만`, `실망`, `비추천`, `파손`, `문제`, `후회`, `아쉬움`, `별로`

### BM25에서의 Expansion 사용

BM25는 exact keyword와 변형 표현에 민감하므로 expansion 효과가 크다. 현재 구현은 다음 clause를 함께 사용한다.

초기 설계:

- 원본 query: high boost
- synonym-expanded query: low-medium boost
- aspect term query: medium boost
- sentiment term query: low boost
- `issue_category` term query: medium-high boost
- `issue_sentiments` nested issue_category+sentiment query: highest boost
- aspect-sentiment positive/negative phrase: medium-high boost
- query의 target issue category와 sentiment가 명확하면 `issue_sentiments` nested filter를 적용

BM25 should clause 예시:

```text
original query over BM25 fields: boost 3.0
expanded synonyms over BM25 fields: boost 1.2
aspect terms over content/title/summary/aspect_keywords: boost 1.6
sentiment terms over content/title/summary: boost 0.8
issue_category exact match: boost 2.2
issue_sentiments.issue_category match: boost 3.0
issue_sentiments.sentiment match: boost 3.0
issue_sentiments.evidence aspect match: boost 2.2
issue_sentiments.evidence sentiment match: boost 1.6
aspect-sentiment phrases: boost 2.4
survey answer query: boost 0.6
```

짧은 query에서는 expansion을 넓히고, 긴 query에서는 원문 matching 비중을 더 높인다. Q1 같은 aspect-sentiment query에서는 `뽁뽁이`와 `만족`이 문서 어딘가에 따로 있는 것보다, `issue_sentiments`에 `배송/포장 + positive`가 저장된 문서만 후보로 남긴다.

### Dense에서의 Expansion 사용

Dense retrieval은 query 의미를 embedding하므로 무작정 synonym을 나열하면 오히려 embedding이 흐려질 수 있다.
다만 query의 target issue category와 sentiment가 명확하면 BM25와 동일하게 `issue_sentiments` nested filter를 dense kNN에도 적용한다.

초기 설계:

- 기본 dense query는 원본 query를 사용한다.
- 긴 query에 한해 intent-preserving rewrite 1개를 추가로 embedding한다.
- dense retrieval 결과는 원본 dense query top 100을 기본으로 하고, rewrite dense query 결과는 보조 candidate로만 합친다.

Dense query text 예시:

```text
원본: 뽁뽁이 포장 상태에 만족한 리뷰
rewrite: 보호 포장이나 완충재가 잘 되어 안전하게 받았다고 만족하는 리뷰
```

### LLM Expansion 사용 조건

LLM expansion은 모든 query에 항상 쓰지 않는다.

사용하는 경우:

- query가 길고 조건이 여러 개일 때
- query의 intent, target sentiment, aspect를 분리해야 할 때
- BM25와 dense 모두 gold recall이 낮은 query를 분석할 때

기본 구현에서는 dictionary expansion만 사용한다. OpenAI rewrite를 실험하려면 `.env`에 다음 값을 둔다.

```text
OPENAI_QUERY_EXPANSION=true
OPENAI_QUERY_EXPANSION_MODEL=gpt-4.1-mini
```

출력 schema:

```json
{
  "rewritten_query": "...",
  "bm25_terms": ["...", "..."],
  "dense_query": "...",
  "must_include": ["...", "..."],
  "should_include": ["...", "..."],
  "should_exclude": ["...", "..."],
  "target_sentiment": "positive|negative|neutral|unknown",
  "target_aspect": "배송/포장|배터리/충전|발열|카메라|성능/속도|디자인/무게|화면/디스플레이|가격/가성비|교환/반품/AS|기타"
}
```

`must_include`는 hard filter가 아니라 reranking/post filtering feature로 먼저 사용한다.

### Expansion Debugging

Expansion은 MAP에 양날의 검이므로 ablation을 반드시 기록한다.

- no expansion
- dictionary expansion only
- dictionary + intent rewrite
- dictionary + LLM expansion

비교 지표:

- BM25 top 100 recall
- Dense top 100 recall
- RRF top 50 recall
- MAP@10

초기 목표는 expansion으로 BM25/Dense recall을 올리되, 최종 MAP@10이 떨어지지 않는 조합을 찾는 것이다.

## Step 1. BM25 Retrieval

BM25는 키워드 일치, 구체적인 표현, 부정/긍정 단서, 제품명/옵션명 일치에 강하다.

사용 필드:

- `content`: 가장 중요한 본문 필드
- `title`: 짧은 핵심 표현이 들어갈 수 있어 boost
- `product_name.search`: 상품명 검색용 text subfield
- `item_name`: 색상, 용량, 자급제 등 옵션 정보
- `short_summary`: LLM enrichment가 있을 때 query와 빠르게 맞는 요약
- `aspect_keywords.search`: 카메라, 배터리, 포장 등 aspect keyword
- `issue_sentiments`: issue_category와 sentiment가 같은 evidence 안에서 맞는지 확인하는 nested field
- `review_survey_answers.answer.search`: 설문 답변 텍스트

초기 field weight 제안:

```text
content^3
title^2
short_summary^2
aspect_keywords.search^2
product_name.search^1.5
item_name^1.2
review_survey_answers.answer.search^1
```

BM25 query 구성:

- 기본은 `multi_match`의 `best_fields` 또는 `most_fields`
- 짧은 키워드 query는 `operator: or`로 recall 확보
- 긴 복합 query는 `minimum_should_match`를 낮게 두어 일부 조건 누락으로 전체 후보가 사라지지 않게 함
- 설문 답변은 `nested` query로 별도 `should`에 포함
- query에서 `target_aspect`와 `target_sentiment`가 잡히면 `issue_sentiments` nested query를 추가해 같은 issue category 안의 감성이 맞는 문서를 boost

BM25 output:

- 최대 100개
- 각 candidate에 `review_id`, BM25 rank, BM25 score 저장

## Step 2. Dense Retrieval

Dense retrieval은 표현이 정확히 일치하지 않아도 의미가 가까운 리뷰를 찾기 위한 단계다.

사용 필드:

- query side: 사용자 query를 그대로 embedding
- document side: `dense_vector`
- document vector source: `dense_text`

Elasticsearch에서는 `knn` 또는 `script_score`를 사용할 수 있다.

초기 설정:

```text
field: dense_vector
k: 100
num_candidates: 300-1000
similarity: cosine
```

Dense retrieval output:

- 최대 100개
- 각 candidate에 `review_id`, dense rank, dense score 저장

## Step 3. RRF Merge

BM25와 dense retrieval 결과를 Reciprocal Rank Fusion으로 합친다.

RRF score:

```text
rrf_score(doc) = sum(1 / (k + rank_i(doc)))
```

초기 설정:

```text
k = 60
bm25_top_k = 100
dense_top_k = 100
rrf_candidate_top_k = 50
```

처리 규칙:

- BM25와 dense 양쪽에 모두 나온 문서는 RRF 점수가 합산된다.
- 한쪽에만 나온 문서도 candidate로 유지한다.
- 같은 `review_id`는 하나로 합친다.
- tie가 있으면 두 retrieval score 중 더 높은 normalized score를 보조 기준으로 쓴다.

RRF output:

- 최대 50개
- 이후 cross-encoder reranking 대상으로 사용

## Step 4. Cross-Encoder Reranking

RRF top 50은 recall 중심 후보이므로, 최종 ranking은 cross-encoder가 query-document relevance를 직접 판단하도록 한다.

Reranker 입력:

```text
query: 사용자 query
document: dense_text 또는 compact rerank text
```

초기 rerank document text는 `dense_text`를 그대로 사용한다. 너무 긴 리뷰가 많아 latency가 크면 다음 compact format으로 줄인다.

```text
상품명: ...
제목: ...
요약: ...
관점 키워드: ...
본문: ...
```

Cross-encoder 후보 모델:

- Korean 또는 multilingual reranker 우선
- 예: `BAAI/bge-reranker-v2-m3`
- 로컬 실행이 어렵거나 느리면 API 기반 reranker 또는 LLM judge로 대체 가능

Reranker output:

- RRF top 50 각각에 rerank score 부여
- rerank score 기준으로 정렬

## Step 5. Post Filtering

Post filtering은 RRF와 cross-encoder reranking 뒤 최종 출력 오류와 중복을 제거한다. Query의 target issue category와 sentiment가 명확하면 최종 결과에서도 `issue_sentiments` 일치 여부를 확인한다.

초기 정책:

- `review_id` 중복 제거
- 본문 fingerprint 중복 제거
- `content`가 비어 있거나 너무 짧은 문서 제거
- target issue category와 sentiment가 명확하면 `issue_sentiments`가 둘 다 맞는 문서만 유지
- 최종 출력에 필요한 `review_id`, `content`, `product_name` 확인
- reranker score 순서를 가능한 유지

중복 fingerprint 예시:

```text
lowercase -> whitespace 제거 -> punctuation 제거 -> 앞 N자 또는 hash
```

최종 top 10은 `final_score` 순으로 반환한다.

후보가 10개 미만으로 줄면 RRF rank가 높은 후보를 보충한다.

추후 ablation 후 고려할 확장:

- query sentiment와 `rating` 방향이 강하게 충돌할 때 약한 penalty
- query aspect가 `issue_category`/`aspect_keywords`와 전혀 맞지 않을 때 약한 penalty
- query에 제품/옵션명이 명시되어 있는데 `product_name`/`item_name`이 다를 때 penalty
- `human_eval.json.intent.exclude`와 직접 충돌하는 표현이 있을 때 penalty

## Scoring Strategy

최종 score는 cross-encoder rerank score를 기본으로 한다.

필요하면 다음 보조 점수를 logging한다.

- `bm25_score`
- `dense_score`
- `rrf_score`
- `rerank_score`
- `filter_penalty`
- `final_score`

초기 final score:

```text
final_score = rerank_score
```

BM25/dense/RRF score는 최종 score에 직접 섞지 않고, reranker 전 후보 생성과 debugging에 주로 사용한다.

## Debugging Plan

`human_eval.json`을 이용해 query별로 다음을 기록한다.

- BM25 top 100 recall
- Dense top 100 recall
- RRF top 50 recall
- RRF top 50 AP upper bound
- Rerank top 10 recall
- Rerank AP@10
- Post-filter top 10 recall
- Post-filter AP@10
- top 10의 review_id 목록
- missing gold review_id
- exclude 조건 위반 문서

우선순위:

1. RRF top 50 안에 gold가 충분히 들어오는지 확인한다.
2. RRF top 50에는 있는데 reranker top 10에서 밀린 gold를 확인한다.
3. reranker top 10에는 있는데 post filtering 후 rank가 내려간 gold를 확인한다.
4. AP@10이 떨어진 query에서 expansion, reranking, filtering 중 어느 단계가 원인인지 분리한다.
5. query category별 MAP@10을 따로 본다.

## Implementation Checklist

- query expansion dictionary 구현
- optional LLM query expansion schema 구현
- batch/checkpoint 기반 index build script 구현
- `build_search_body()`에 BM25 retrieval query 구현
- dense query embedding 생성
- Elasticsearch dense retrieval top 100 구현
- BM25 결과와 dense 결과를 `review_id` 기준으로 merge
- RRF 함수 구현
- RRF top 50 source fetch 정리
- cross-encoder reranker wrapper 구현
- post filtering 함수 구현
- `search()`에서 전체 pipeline 연결
- `human_eval.json` 기반 MAP@10 debugging script 추가

## Index Build Workflow

LLM enrichment를 포함한 전체 색인은 오래 걸리고 중간 실패 가능성이 있으므로 batch 단위로 처리한다.

기본 스크립트:

```bash
uv run python task2_retrieval/build_index.py --llm --recreate
```

권장 smoke test:

```bash
uv run python task2_retrieval/build_index.py \
  --limit 1 \
  --llm \
  --recreate \
  --index-name reviews_llm_smoke
```

중간 저장 파일:

- 성공 document: `output/task2_retrieval/indexed_documents.jsonl`
- 실패 record/error: `output/task2_retrieval/index_failures.jsonl`

batch 동작:

- record를 하나씩 preprocess한다.
- `--llm`이면 OpenAI enrichment를 적용한다.
- 성공한 document는 batch 단위로 checkpoint JSONL에 append한다.
- 같은 batch를 Elasticsearch bulk API로 색인한다.
- 실패한 record는 Elasticsearch에 올리지 않고 failed log에 남긴다.

중단 후 이어서 처리:

```bash
uv run python task2_retrieval/build_index.py --llm --resume
```

checkpoint만 만들고 Elasticsearch에는 올리지 않기:

```bash
uv run python task2_retrieval/build_index.py --llm --no-index
```

이미 만들어진 checkpoint를 Elasticsearch에 다시 올리기:

```bash
uv run python task2_retrieval/build_index.py \
  --index-from-checkpoint \
  --recreate
```

`--recreate`는 index를 삭제 후 다시 만들기 때문에, 실제 index에 전체 재색인을 할 때만 사용한다.

## Initial Defaults

```text
QUERY_EXPANSION = dictionary
BM25_TOP_K = 100
DENSE_TOP_K = 100
RRF_K = 60
RRF_CANDIDATE_TOP_K = 50
FINAL_TOP_K = 10
```

필드 기본값:

```text
BM25 fields:
- content^3
- title^2
- short_summary^2
- aspect_keywords.search^2
- product_name.search^1.5
- item_name^1.2
- review_survey_answers.answer.search^1
- issue_sentiments nested match for issue_category/sentiment/evidence

Dense fields:
- query embedding: query
- document embedding source: dense_text
- Elasticsearch vector field: dense_vector
```
