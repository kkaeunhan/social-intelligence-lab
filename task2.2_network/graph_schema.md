# Graph Schema Design

## 목적

`review_for_analysis.json`의 쿠팡 전자제품 리뷰를 Product, Review, Keyword, Sentiment 중심의 그래프로 저장하고, 리뷰 안에서 함께 등장한 키워드쌍의 co-occurrence, PMI, NPMI를 관계 속성으로 보존한다.

스키마는 Step 1의 필수 요구인 `Product`, `Review`, `Keyword`, `Sentiment` 노드와 `HAS_REVIEW`, `MENTIONS`, `CO_OCCURS_WITH` 관계를 기본으로 한다. Step 6의 BI 해석 결과를 저장할 수 있도록 `ContextLabel`, `BusinessAction` 노드는 확장 영역으로 둔다.

## 데이터 관찰 반영

입력 파일: `si_dataset/review_for_analysis.json`

- 리뷰 수: 2,757
- 상품 그룹 필드: `product_name`
- 상품 상세/옵션 필드: `productId`, `vendorItemId`, `itemId`, `itemName`
- 리뷰 식별자: `reviewId`
- 텍스트 원천: `title`, `content`, `reviewSurveyAnswers[].question`, `reviewSurveyAnswers[].answer`
- 평점: `rating` 1-5
- 설문 문항: `가성비`, `사용 시간`, `디자인`, `카메라 성능`, `무게`
- 기존 `task2_retrieval` 전처리는 `title`, `content`, 설문 응답을 보존하고, HTML/URL/email/전화번호/과도한 반복 문자를 정규화한다.

따라서 Step 2 이후의 분석 텍스트는 원문 필드를 보존하면서 아래처럼 구성한다.

```text
제목: {title}
본문: {content}
설문: {question}: {answer} / ...
```

## Constraint 속성명 확인

중복 노드 생성을 막는 unique constraint는 반드시 필요하다. 다만 원본 JSON의 camelCase 필드를 그래프에 그대로 쓰지 않고, Step 2 정제 산출물에서는 snake_case를 canonical 속성명으로 사용한다.

- 원본 `reviewId` -> 그래프/정제 산출물 `review_id`
- 키워드 표시명 -> 그래프 `Keyword.text`, unique key는 `keyword_id`
- 감성 표시명 `label`은 한국어 표시값이고, unique key는 `sentiment_id`

따라서 Neo4j 제약은 `Product.product_name`, `Review.review_id`, `Keyword.keyword_id`, `Sentiment.sentiment_id` 기준으로 생성한다.

## Node Labels

### Product

상품 그룹 또는 상세 상품을 나타낸다. 제품별 BI 비교를 위해 `product_name`을 1차 식별자로 사용하고, 상세 옵션은 Review 쪽에도 보존한다.

Primary key:

- `product_name`

Properties:

- `product_name`: 예: `iphone_17_pro`
- `display_name`: 분석/리포트 표시용 이름, 초기값은 `product_name`
- `product_family`: 선택. 예: `iphone`, `galaxy_s`, `galaxy_z`
- `brand`: 선택. 예: `Apple`, `Samsung`
- `review_count`: 적재 후 집계값

### Review

개별 쿠팡 리뷰를 나타낸다. 원문 텍스트, 정제 텍스트, 평점, 도움돼요 수, 첨부 여부를 저장한다.

Primary key:

- `review_id`

Properties:

- `review_id`: `reviewId` 문자열화
- `review_at`: `reviewAt` epoch millis
- `created_at`: `createdAt` epoch millis
- `product_id`: `productId`
- `vendor_item_id`: `vendorItemId`
- `item_id`: `itemId`
- `item_name`: `itemName`
- `title`: 정제된 제목
- `content`: 정제된 본문
- `survey_text`: `질문: 답변 / ...`
- `analysis_text`: 제목, 본문, 설문을 통합한 분석용 텍스트
- `analysis_char_len`: 분석용 텍스트 길이
- `rating`: 1-5
- `helpful_count`: `helpfulCount`
- `helpful_true_count`: `helpfulTrueCount`
- `helpful_false_count`: `helpfulFalseCount`
- `has_image`: 이미지 첨부 여부
- `has_video`: 동영상 첨부 여부
- `keyword_count`: 추출된 키워드 수

### Keyword

리뷰에서 추출된 제품 인식 키워드다. 명사, 형용사, 동사 기반 키워드를 모두 수용하되, 표제어/정규화 형태를 기준으로 병합한다.

Primary key:

- `keyword_id`

Properties:

- `keyword_id`: 정규화 키워드 기반 안정 ID. 예: `kw:배터리`
- `text`: 표시 텍스트. 예: `배터리`
- `normalized`: 정규화 텍스트. 예: `배터리`
- `pos_group`: `noun`, `adjective`, `verb`, `phrase`, `unknown`
- `df`: 키워드가 등장한 리뷰 수
- `tf`: 전체 등장 빈도
- `product_df`: 선택. 상품별 df는 관계 속성 또는 별도 집계 파일에도 저장 가능
- `is_stopword`: 불용어 여부

### Sentiment

평점 기반 감성 구간이다. Step 4 이후 긍정/부정 리뷰에서 연관어가 다르게 나타나는지 비교하기 위한 기준 노드로 사용한다.

Primary key:

- `sentiment_id`

Properties:

- `sentiment_id`: `positive`, `neutral`, `negative`
- `label`: 표시명. 예: `긍정`
- `rating_min`
- `rating_max`
- `rule`: 감성 분류 규칙 설명

초기 규칙:

- `positive`: rating 4-5
- `neutral`: rating 3
- `negative`: rating 1-2

리뷰 분포가 5점에 크게 치우쳐 있으므로, 이후 분석에서는 `positive` 내부를 `strong_positive=5`, `weak_positive=4`로 추가 분해할 수 있다. Step 1에서는 BI 비교의 기본 축을 단순하고 해석 가능한 3구간으로 둔다.

### ContextLabel

Step 6에서 LLM이 연관어쌍 또는 subgraph에 부여한 고객 인식 맥락 라벨이다.

Primary key:

- `context_id`

Properties:

- `context_id`
- `label`: 예: `배송 포장 불안`, `카메라 기대 미달`
- `summary`
- `evidence`
- `model`
- `created_at`

### BusinessAction

ContextLabel을 마케팅, 제품 개선, CS 대응 액션으로 연결한다.

Primary key:

- `action_id`

Properties:

- `action_id`
- `action_type`: `marketing`, `product_improvement`, `cs`
- `title`
- `description`
- `priority`: `high`, `medium`, `low`

## Relationships

### (:Product)-[:HAS_REVIEW]->(:Review)

상품과 리뷰의 소속 관계다.

Properties:

- `review_at`
- `rating`

### (:Review)-[:HAS_SENTIMENT]->(:Sentiment)

리뷰 평점 기반 감성 구간이다.

Properties:

- `rating`
- `rule_version`: 예: `rating_v1`

### (:Review)-[:MENTIONS]->(:Keyword)

개별 리뷰가 특정 키워드를 언급했음을 나타낸다.

Properties:

- `count`: 해당 리뷰 내 등장 횟수
- `positions`: 선택. 토큰 위치 배열
- `sources`: `title`, `content`, `survey` 중 등장 출처 배열
- `weight`: 기본 `1.0`, 제목/설문 가중치 적용 시 조정

### (:Keyword)-[:CO_OCCURS_WITH]->(:Keyword)

같은 리뷰 안에서 함께 등장한 키워드쌍의 연관 관계다. 무방향 의미지만 Neo4j에는 정렬된 방향으로 한 번만 저장한다.

생성 규칙:

- 한 리뷰 내 unique keyword set에서 unordered pair 생성
- `keyword_a.normalized < keyword_b.normalized`인 방향으로 저장
- 자기 자신 쌍은 제외
- 상품별/감성별 비교가 필요하므로 관계 속성에 집계 scope를 명시하거나, scope별 관계를 따로 생성한다.

Properties:

- `scope`: `global`, `product`, `sentiment`, `product_sentiment`
- `product_name`: scope가 상품을 포함할 때만 사용
- `sentiment_id`: scope가 감성을 포함할 때만 사용
- `co_count`: 두 키워드가 함께 등장한 리뷰 수
- `keyword_a_count`: scope 내 keyword A 리뷰 수
- `keyword_b_count`: scope 내 keyword B 리뷰 수
- `review_count`: scope 내 전체 리뷰 수
- `pmi`
- `npmi`
- `confidence`: `co_count / keyword_a_count`
- `lift`
- `updated_at`

PMI/NPMI 계산 기준:

```text
P(a) = keyword_a_count / review_count
P(b) = keyword_b_count / review_count
P(a,b) = co_count / review_count
PMI(a,b) = log2(P(a,b) / (P(a) * P(b)))
NPMI(a,b) = PMI(a,b) / -log2(P(a,b))
```

초기 필터 권장값:

- `co_count >= 3`
- `npmi > 0`
- 너무 일반적인 키워드는 Step 3의 불용어/문서빈도 상한으로 제거

### (:CO_OCCURS_WITH)-level Evidence

Neo4j 관계는 관계에서 Review로 직접 관계를 만들기 어렵다. 따라서 근거 리뷰는 다음 중 하나로 저장한다.

- 관계 속성 `sample_review_ids`: 상위 근거 리뷰 ID 배열
- 별도 `CoOccurrence` 노드 도입

Step 1 기본 스키마는 단순성을 위해 `CO_OCCURS_WITH.sample_review_ids`를 사용한다. 이후 근거 탐색이 중요해지면 `(:CoOccurrence)` 노드를 추가한다.

### (:ContextLabel)-[:EXPLAINS]->(:Keyword)

맥락 라벨이 특정 키워드를 설명한다.

Properties:

- `role`: `primary`, `secondary`

### (:ContextLabel)-[:EXPLAINS_PAIR]->(:Keyword)

Neo4j에서 pair 자체를 노드로 두지 않는 기본안에서는 `ContextLabel`이 두 Keyword에 각각 연결되고, `keyword_pair_key` 속성으로 한 쌍을 식별한다.

Properties:

- `keyword_pair_key`: 예: `배터리||오래감`
- `product_name`
- `sentiment_id`
- `npmi`
- `co_count`

### (:ContextLabel)-[:MAPS_TO]->(:BusinessAction)

맥락 라벨과 BI 액션의 연결이다.

Properties:

- `rationale`
- `priority`

## 권장 탐색 쿼리

제품별 핵심 인식 키워드:

```cypher
MATCH (:Product {product_name: $product_name})-[:HAS_REVIEW]->(:Review)-[m:MENTIONS]->(k:Keyword)
RETURN k.text AS keyword, count(*) AS review_df, sum(m.count) AS tf
ORDER BY review_df DESC, tf DESC
LIMIT 30;
```

제품별 상위 NPMI 연관어:

```cypher
MATCH (a:Keyword)-[r:CO_OCCURS_WITH {scope: "product", product_name: $product_name}]->(b:Keyword)
WHERE r.co_count >= 3 AND r.npmi > 0
RETURN a.text AS keyword_a, b.text AS keyword_b, r.co_count AS co_count, r.pmi AS pmi, r.npmi AS npmi
ORDER BY r.npmi DESC, r.co_count DESC
LIMIT 50;
```

부정 리뷰에서 강한 연관어:

```cypher
MATCH (a:Keyword)-[r:CO_OCCURS_WITH {scope: "product_sentiment", product_name: $product_name, sentiment_id: "negative"}]->(b:Keyword)
WHERE r.co_count >= 3 AND r.npmi > 0
RETURN a.text AS keyword_a, b.text AS keyword_b, r.co_count AS co_count, r.npmi AS npmi
ORDER BY r.npmi DESC, r.co_count DESC
LIMIT 50;
```

연관어쌍 근거 리뷰:

```cypher
MATCH (a:Keyword {normalized: $keyword_a})-[r:CO_OCCURS_WITH]->(b:Keyword {normalized: $keyword_b})
WITH r.sample_review_ids AS ids
MATCH (rv:Review)
WHERE rv.review_id IN ids
RETURN rv.review_id, rv.rating, rv.title, rv.content
LIMIT 10;
```

## 설계 결정

- Product는 `product_name` 기준으로 먼저 묶는다. 데이터에 상세 옵션이 많기 때문에, 옵션 분석은 Review 속성의 `item_name`, `item_id`, `vendor_item_id`로 내려가서 본다.
- Sentiment는 별도 노드로 둔다. 평점 기반 감성별 subgraph 탐색이 목적에 포함되어 있고, 이후 LLM issue sentiment를 추가해도 확장하기 쉽다.
- CO_OCCURS_WITH는 Keyword 간 관계로 둔다. PMI/NPMI 자체가 키워드쌍의 속성이므로 그래프 탐색이 가장 간단하다.
- 단, relationship key uniqueness가 Neo4j에서 제약으로 강제하기 어렵기 때문에 적재 코드에서 `pair_key`, `scope`, `product_name`, `sentiment_id` 기준으로 MERGE해야 한다.
- Step 2 전처리는 `task2_retrieval.preprocess_record`의 정규화 규칙을 최대한 재사용하되, 네트워크 분석에서는 `content`뿐 아니라 `title`과 `survey_text`까지 통합한 `analysis_text` 길이로 필터링한다.
