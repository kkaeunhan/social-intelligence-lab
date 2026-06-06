// Neo4j schema for task2.2_network
// Run after creating/choosing the target database.

CREATE CONSTRAINT product_product_name IF NOT EXISTS
FOR (p:Product)
REQUIRE p.product_name IS UNIQUE;

CREATE CONSTRAINT review_review_id IF NOT EXISTS
FOR (r:Review)
REQUIRE r.review_id IS UNIQUE;

CREATE CONSTRAINT keyword_keyword_id IF NOT EXISTS
FOR (k:Keyword)
REQUIRE k.keyword_id IS UNIQUE;

CREATE CONSTRAINT sentiment_sentiment_id IF NOT EXISTS
FOR (s:Sentiment)
REQUIRE s.sentiment_id IS UNIQUE;

CREATE CONSTRAINT context_context_id IF NOT EXISTS
FOR (c:ContextLabel)
REQUIRE c.context_id IS UNIQUE;

CREATE CONSTRAINT action_action_id IF NOT EXISTS
FOR (a:BusinessAction)
REQUIRE a.action_id IS UNIQUE;

CREATE INDEX context_issue_type IF NOT EXISTS
FOR (c:ContextLabel)
ON (c.issue_type);

CREATE INDEX context_product_name IF NOT EXISTS
FOR (c:ContextLabel)
ON (c.product_name);

CREATE INDEX context_sentiment_id IF NOT EXISTS
FOR (c:ContextLabel)
ON (c.sentiment_id);

CREATE INDEX context_confidence IF NOT EXISTS
FOR (c:ContextLabel)
ON (c.confidence);

CREATE INDEX action_action_type IF NOT EXISTS
FOR (a:BusinessAction)
ON (a.action_type);

CREATE INDEX action_priority IF NOT EXISTS
FOR (a:BusinessAction)
ON (a.priority);

CREATE INDEX review_rating IF NOT EXISTS
FOR (r:Review)
ON (r.rating);

CREATE INDEX review_review_at IF NOT EXISTS
FOR (r:Review)
ON (r.review_at);

CREATE INDEX keyword_normalized IF NOT EXISTS
FOR (k:Keyword)
ON (k.normalized);

CREATE INDEX keyword_pos_group IF NOT EXISTS
FOR (k:Keyword)
ON (k.pos_group);

CREATE INDEX co_occurs_scope IF NOT EXISTS
FOR ()-[r:CO_OCCURS_WITH]-()
ON (r.scope);

CREATE INDEX co_occurs_product_name IF NOT EXISTS
FOR ()-[r:CO_OCCURS_WITH]-()
ON (r.product_name);

CREATE INDEX co_occurs_sentiment_id IF NOT EXISTS
FOR ()-[r:CO_OCCURS_WITH]-()
ON (r.sentiment_id);

CREATE INDEX co_occurs_npmi IF NOT EXISTS
FOR ()-[r:CO_OCCURS_WITH]-()
ON (r.npmi);

MERGE (s:Sentiment {sentiment_id: "positive"})
SET s.label = "긍정",
    s.rating_min = 4,
    s.rating_max = 5,
    s.rule = "rating 4-5";

MERGE (s:Sentiment {sentiment_id: "neutral"})
SET s.label = "중립",
    s.rating_min = 3,
    s.rating_max = 3,
    s.rule = "rating 3";

MERGE (s:Sentiment {sentiment_id: "negative"})
SET s.label = "부정",
    s.rating_min = 1,
    s.rating_max = 2,
    s.rule = "rating 1-2";
