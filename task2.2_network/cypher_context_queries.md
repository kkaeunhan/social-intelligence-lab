# Step 6 Context Label Queries

LLM 기반 ContextLabel, BusinessAction 적재 후 Neo4j Browser 또는 `cypher-shell`에서 확인할 쿼리입니다.

## Context/Action 적재 수 확인

```cypher
CALL () {
  MATCH (c:ContextLabel) RETURN "ContextLabel" AS name, count(c) AS count
  UNION ALL
  MATCH (a:BusinessAction) RETURN "BusinessAction" AS name, count(a) AS count
  UNION ALL
  MATCH ()-[r:HAS_CONTEXT]->() RETURN "HAS_CONTEXT" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:SUPPORTED_BY]->() RETURN "SUPPORTED_BY" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:EXPLAINS_PAIR]->() RETURN "EXPLAINS_PAIR" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:MAPS_TO]->() RETURN "MAPS_TO" AS name, count(r) AS count
}
RETURN name, count
ORDER BY name;
```

## Issue Type 분포 확인

```cypher
MATCH (c:ContextLabel)
RETURN c.issue_type AS issue_type, count(c) AS context_count
ORDER BY context_count DESC, issue_type;
```

## 제품별 핵심 Context 확인

```cypher
MATCH (:Product {product_name: $product_name})-[:HAS_CONTEXT]->(c:ContextLabel)
RETURN
  c.sentiment_id AS sentiment_id,
  c.issue_type AS issue_type,
  c.label AS context_label,
  c.summary AS summary,
  c.confidence AS confidence,
  c.max_npmi AS max_npmi,
  c.total_pair_count AS total_pair_count
ORDER BY c.sentiment_id, c.max_significance_score DESC
LIMIT 30;
```

Example parameter:

```cypher
:param product_name => "iphone_17_pro";
```

## 제품 기능 이슈만 보기

```cypher
MATCH (:Product {product_name: $product_name})-[:HAS_CONTEXT]->(c:ContextLabel)
WHERE c.issue_type = "product_feature"
MATCH (c)-[:MAPS_TO]->(a:BusinessAction)
RETURN
  c.sentiment_id AS sentiment_id,
  c.label AS context_label,
  c.customer_perception AS customer_perception,
  a.action_type AS action_type,
  a.title AS action_title,
  a.priority AS priority
ORDER BY c.max_significance_score DESC, priority;
```

## 배송/포장/구매 경험 이슈만 보기

```cypher
MATCH (:Product {product_name: $product_name})-[:HAS_CONTEXT]->(c:ContextLabel)
WHERE c.issue_type = "purchase_delivery_experience"
MATCH (c)-[:MAPS_TO]->(a:BusinessAction)
RETURN
  c.sentiment_id AS sentiment_id,
  c.label AS context_label,
  c.customer_perception AS customer_perception,
  a.action_type AS action_type,
  a.title AS action_title,
  a.priority AS priority
ORDER BY c.max_significance_score DESC, priority;
```

## 근거 리뷰까지 추적

```cypher
MATCH (:Product {product_name: $product_name})-[:HAS_CONTEXT]->(c:ContextLabel)-[:SUPPORTED_BY]->(r:Review)
WHERE c.issue_type = $issue_type
RETURN
  c.label AS context_label,
  r.review_id AS review_id,
  r.rating AS rating,
  r.title AS title,
  left(r.content, 220) AS content_excerpt
ORDER BY c.max_significance_score DESC, r.helpful_count DESC
LIMIT 30;
```

Example parameters:

```cypher
:param product_name => "iphone_17_pro";
:param issue_type => "purchase_delivery_experience";
```
