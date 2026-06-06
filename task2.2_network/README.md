# Task 2.2 Network Analysis

쿠팡 전자제품 리뷰에서 고객이 제품을 어떤 단어와 함께 인식하는지 분석하기 위한 Graph DB 및 PMI/NPMI 기반 연관어 네트워크 작업 공간입니다.

## 진행 순서

1. Graph schema 설계
2. 리뷰 데이터 정제
3. 키워드 추출
4. PMI/NPMI 연관어 계산
5. Neo4j Graph DB 적재
6. LLM 기반 맥락 해석

## 현재 산출물

- `graph_schema.md`: Step 1 Graph schema 설계 문서
- `cypher/schema.cypher`: Neo4j 제약 조건과 인덱스 정의
- `preprocess_reviews.py`: Step 2 리뷰 정제 및 분석용 텍스트 생성 스크립트
- `extract_keywords.py`: Step 3 키워드 추출 및 Review-Keyword 관계 생성 스크립트
- `calculate_pmi_npmi.py`: Step 4 co-occurrence, PMI, NPMI 계산 스크립트
- `load_to_neo4j.py`: Step 5 Neo4j Graph DB 적재 스크립트
- `cypher_queries.md`: Step 5 검증 및 기본 분석 Cypher 쿼리 모음
- `generate_context_candidates.py`: Step 6 LLM 해석 전 context insight 후보 생성 스크립트
- `label_context_clusters.py`: Step 6 LLM 기반 context labeling 스크립트
- `load_context_to_neo4j.py`: Step 6 ContextLabel/BusinessAction Neo4j 적재 스크립트
- `cypher_context_queries.md`: Step 6 ContextLabel/BusinessAction 확인 쿼리 모음

## Step 2 실행

```bash
python task2.2_network/preprocess_reviews.py
```

생성 파일:

- `task2.2_network/data/cleaned_reviews.jsonl`
- `task2.2_network/data/cleaned_reviews_sample.json`
- `task2.2_network/data/preprocess_summary.json`

## Step 3 실행

```bash
uv run python task2.2_network/extract_keywords.py --analyzer kiwi
```

생성 파일:

- `task2.2_network/data/keywords.jsonl`
- `task2.2_network/data/review_keywords.jsonl`
- `task2.2_network/data/review_keyword_edges.jsonl`
- `task2.2_network/data/keyword_extraction_summary.json`

## Step 4 실행

```bash
python task2.2_network/calculate_pmi_npmi.py
```

기본 필터:

- 리뷰별 keyword `top_n=20`
- `min_doc_freq=5`
- `max_doc_ratio=0.4`
- `min_pair_count=3`

생성 파일:

- `task2.2_network/data/network_review_keywords.jsonl`
- `task2.2_network/data/co_occurrence_edges.jsonl`
- `task2.2_network/data/pmi_npmi_summary.json`

## Step 5 실행

`.env`에 아래 값이 있어야 합니다.

```text
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

Neo4j 컨테이너가 실행 중인 상태에서:

```bash
uv run python task2.2_network/load_to_neo4j.py
```

기존 Step 5 데이터를 지우고 현재 산출물 기준으로 다시 적재하려면:

```bash
uv run python task2.2_network/load_to_neo4j.py --reset
```

대량 관계 적재가 부담되면 smoke test를 먼저 실행할 수 있습니다.

```bash
uv run python task2.2_network/load_to_neo4j.py --limit 100 --skip-co-occurrence
```

생성 파일:

- `task2.2_network/data/neo4j_load_summary.json`

기본 확인 쿼리는 `task2.2_network/cypher_queries.md`에 정리되어 있습니다.

## Step 6 Pre-LLM 실행

LLM 해석 전에 제품/감성/테마별 insight 후보와 원문 근거 패키지를 생성합니다.

```bash
uv run python task2.2_network/generate_context_candidates.py
```

기본 필터:

- `min_count=5`
- `min_npmi=0.2`
- 제품-감성별 상위 후보 `top_product_sentiment=8`
- 기타 scope별 상위 후보 `top_per_group=12`
- 후보별 원문 근거 `sample_reviews=3`

생성 파일:

- `task2.2_network/data/context_insight_candidates.jsonl`
- `task2.2_network/data/context_candidate_clusters.jsonl`
- `task2.2_network/data/context_candidate_summary.json`
- `task2.2_network/data/context_insight_candidates.md`

## Step 6 LLM Context Labeling 실행

`context_candidate_clusters.jsonl`을 입력으로 사용해 고정 JSON 형식의 context label과 business action을 생성합니다.

```bash
uv run python task2.2_network/label_context_clusters.py
```

생성 파일:

- `task2.2_network/data/context_labels.jsonl`
- `task2.2_network/data/context_labels_product_feature.jsonl`
- `task2.2_network/data/context_labels_purchase_delivery_experience.jsonl`
- `task2.2_network/data/context_labels_other.jsonl`
- `task2.2_network/data/context_label_summary.json`

Neo4j에 ContextLabel, BusinessAction 노드와 관계를 적재하려면:

```bash
uv run python task2.2_network/load_context_to_neo4j.py --reset-context
```

생성 파일:

- `task2.2_network/data/context_neo4j_load_summary.json`

기본 확인 쿼리는 `task2.2_network/cypher_context_queries.md`에 정리되어 있습니다.

## 입력 데이터

- `../si_dataset/review_for_analysis.json`

현재 데이터는 2,757개 리뷰이며, 상품 그룹은 다음 7개입니다.

- `galaxy_s26`
- `galaxy_s26_ultra`
- `galaxy_z_flip7`
- `galaxy_z_fold7`
- `iphone_17`
- `iphone_17_pro`
- `iphone_17_pro_max`
