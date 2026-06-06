# Neo4j Analysis Queries

Step 5 적재 후 Neo4j Browser 또는 `cypher-shell`에서 실행할 검증/분석 쿼리입니다.

## 전체 노드/관계 수 확인

```cypher
CALL () {
  MATCH (n:Product) RETURN "Product" AS name, count(n) AS count
  UNION ALL
  MATCH (n:Review) RETURN "Review" AS name, count(n) AS count
  UNION ALL
  MATCH (n:Keyword) RETURN "Keyword" AS name, count(n) AS count
  UNION ALL
  MATCH (n:Sentiment) RETURN "Sentiment" AS name, count(n) AS count
  UNION ALL
  MATCH ()-[r:HAS_REVIEW]->() RETURN "HAS_REVIEW" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:HAS_SENTIMENT]->() RETURN "HAS_SENTIMENT" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:MENTIONS]->() RETURN "MENTIONS" AS name, count(r) AS count
  UNION ALL
  MATCH ()-[r:CO_OCCURS_WITH]->() RETURN "CO_OCCURS_WITH" AS name, count(r) AS count
}
RETURN name, count
ORDER BY name;
```

## 제품별 리뷰 수 확인

```cypher
MATCH (p:Product)-[:HAS_REVIEW]->(r:Review)
RETURN p.product_name AS product_name, count(r) AS review_count
ORDER BY review_count DESC;
```

## 제품별 상위 연관어쌍 확인

```cypher
MATCH (a:Keyword)-[r:CO_OCCURS_WITH {scope: "product", product_name: $product_name}]->(b:Keyword)
WHERE r.count >= 3 AND r.npmi > 0
RETURN
  a.text AS keyword_1,
  b.text AS keyword_2,
  r.count AS count,
  r.pmi AS pmi,
  r.npmi AS npmi,
  r.lift AS lift
ORDER BY r.npmi DESC, r.count DESC
LIMIT 30;
```

Example parameter:

```cypher
:param product_name => "iphone_17";
```

## negative 리뷰에서 NPMI가 높은 연관어쌍 확인

```cypher
MATCH (a:Keyword)-[r:CO_OCCURS_WITH {scope: "product_sentiment", product_name: $product_name, sentiment_id: "negative"}]->(b:Keyword)
WHERE r.count >= 3 AND r.npmi > 0
RETURN
  a.text AS keyword_1,
  b.text AS keyword_2,
  r.count AS count,
  r.pmi AS pmi,
  r.npmi AS npmi,
  r.sample_review_ids AS sample_review_ids
ORDER BY r.npmi DESC, r.count DESC
LIMIT 30;
```

## 특정 키워드와 강하게 연결된 키워드 확인

```cypher
MATCH (k:Keyword {normalized: $keyword})-[r:CO_OCCURS_WITH]-(other:Keyword)
WHERE r.scope = "global" AND r.count >= 3 AND r.npmi > 0
RETURN
  other.text AS connected_keyword,
  r.scope AS scope,
  r.product_name AS product_name,
  r.sentiment_id AS sentiment_id,
  r.count AS count,
  r.pmi AS pmi,
  r.npmi AS npmi
ORDER BY r.npmi DESC, r.count DESC
LIMIT 30;
```

Example parameter:

```cypher
:param keyword => "카메라";
```

## 제품-감성별 키워드 빈도 확인

```cypher
MATCH (:Product {product_name: $product_name})-[:HAS_REVIEW]->(r:Review)-[m:MENTIONS]->(k:Keyword)
WHERE r.sentiment_id = $sentiment_id
RETURN
  k.text AS keyword,
  k.pos_group AS pos_group,
  count(DISTINCT r) AS review_df,
  sum(m.count) AS mention_count
ORDER BY review_df DESC, mention_count DESC
LIMIT 30;
```

Example parameters:

```cypher
:param product_name => "iphone_17";
:param sentiment_id => "negative";
```
