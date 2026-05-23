import asyncio
import json
import os
import re
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 100
MIN_SCORE = 60
SEM = asyncio.Semaphore(30)
TIMEOUT = aiohttp.ClientTimeout(total=60)
MAX_API_RETRIES = 2
RATE_LIMIT_RETRY_SECONDS = int(os.getenv("RATE_LIMIT_RETRY_SECONDS", "70"))
CHUNK_DELAY_SECONDS = 10

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GEMINI_MODEL = "gemini-2.0-flash"
USE_GEMINI = os.getenv("USE_GEMINI", "0") == "1"
MODEL_NAMES = [
    name.strip()
    for name in os.getenv("MODEL_NAMES", "openai").split(",")
    if name.strip()
]
if USE_GEMINI and "gemini" not in MODEL_NAMES:
    MODEL_NAMES.append("gemini")

QUERIES_PATH = Path("task2_retrieval/queries_final.json")
REVIEWS_PATH = Path("si_dataset/review_for_analysis.json")
CHECKPOINT_PATH = Path("task2_retrieval/final_evaluation_checkpoint_2.json")
OUTPUT_PATH = Path("task2_retrieval/final_evaluation_report_2.json")

QUERY_INTENTS = {
    "Q01": {
        "intent_type": "positive_topic",
        "target_sentiment": "positive",
        "must_match": ["bubble wrap or protective packaging", "satisfaction with packaging condition"],
        "exclude": ["missing bubble wrap", "damaged box", "poor packaging", "complaint"],
    },
    "Q02": {
        "intent_type": "negative_topic",
        "target_sentiment": "negative",
        "must_match": [
            "the complaint must be specifically about white/whitish color, color tone, yellowish white, discoloration, stains/dirt on white, or choosing white reluctantly",
            "negative feeling must be about that white/color issue, not about delivery, packaging, price, battery, or other topics"
        ],
        "exclude": [
            "positive white color review such as white is pretty, clean, luxurious, satisfying, or no regret",
            "packaging/delivery complaint where white is only the product option",
            "generic color praise or unrelated color comments",
            "complaint not about white/color itself"
        ],
    },
    "Q03": {
        "intent_type": "purchase_context",
        "target_sentiment": "any",
        "must_match": [
            "review says the product was bought for or given to a child, son, daughter, kid, student, or children",
            "explicit recipient context such as child uses it, bought for son's/daughter's phone, or gave it to kids"
        ],
        "exclude": [
            "generic family mention without purchase-for-child context",
            "child is mentioned only as part of an unrelated story"
        ],
    },
    "Q04": {
        "intent_type": "negative_service",
        "target_sentiment": "negative",
        "must_match": [
            "Coupang customer service, seller response, exchange, return, refund, cancellation, pickup, or AS process",
            "dissatisfaction, delay, refusal, inconvenience, repeated contact, or complaint about that process"
        ],
        "exclude": [
            "smooth exchange or return",
            "generic product complaint without service/exchange/return/refund/customer-center context"
        ],
    },
    "Q05": {
        "intent_type": "purchase_context",
        "target_sentiment": "any",
        "must_match": [
            "purchased as a gift, present, birthday gift, parents' gift, family gift, friend gift, spouse/partner gift, or acquaintance gift",
            "explicit recipient context such as mom, dad, parents, husband, wife, girlfriend, boyfriend, friend, family, child, or colleague"
        ],
        "exclude": [
            "self purchase only",
            "generic recommendation without gift/present/recipient context"
        ],
    },
    "Q06": {
        "intent_type": "negative_after_use",
        "target_sentiment": "negative",
        "must_match": [
            "after using for some time, several days/weeks/months, long-term use, or real-use experience",
            "battery drain, short battery life, charging issue, heat, overheating, gets hot, thermal discomfort, or performance drop due to heat"
        ],
        "exclude": [
            "first impression only with no actual usage period",
            "positive battery or heat comment",
            "battery/heat mentioned only as good"
        ],
    },
    "Q07": {
        "intent_type": "negative_value",
        "target_sentiment": "negative",
        "must_match": [
            "price-performance, value for money, expensive for what it offers, not worth the price, overpriced, or cost-performance complaint",
            "performance, quality, specs, features, or satisfaction is disappointing compared with the price"
        ],
        "exclude": [
            "good value for money",
            "price complaint without performance/value/spec/feature comparison"
        ],
    },
    "Q08": {
        "intent_type": "comparison",
        "target_sentiment": "negative",
        "must_match": [
            "comparison with previous model, older model, previous generation, earlier phone, or prior product",
            "little change, no major difference, not much improved, similar, same as before, or upgrade feels minor"
        ],
        "exclude": [
            "says upgrade is large",
            "no previous-model/older-device comparison"
        ],
    },
    "Q09": {
        "intent_type": "purchase_reason",
        "target_sentiment": "any",
        "must_match": [
            "selected high-storage/high-capacity model such as 256GB, 512GB, 1TB, larger capacity, or more storage",
            "reason is photo, video, camera shooting, filming, recording, saving media, or storage for pictures/videos"
        ],
        "exclude": [
            "high capacity without photo/video/media storage reason",
            "photo/video comment without capacity choice"
        ],
    },
    "Q10": {
        "intent_type": "positive_topic",
        "target_sentiment": "positive",
        "must_match": [
            "display, screen, panel, brightness, clarity, resolution, vividness, refresh rate, smoothness, scrolling, or motion",
            "clear positive satisfaction such as very clear, smooth, vivid, bright, good screen, or satisfying display"
        ],
        "exclude": [
            "display complaint",
            "generic product satisfaction without display/screen/smoothness/clarity context"
        ],
    },
    "Q11": {
        "intent_type": "contrast_rating",
        "target_sentiment": "mixed",
        "must_match": [
            "positive product evaluation such as product is good, works well, satisfied with product, or item itself is fine",
            "low star rating, one/two/three stars, rating lowered, gave low score, or explicitly says the score is low for another reason"
        ],
        "exclude": [
            "high rating",
            "only negative product evaluation without product-positive/rating-low contrast"
        ],
    },
    "Q12": {
        "intent_type": "contrast",
        "target_sentiment": "mixed",
        "must_match": [
            "design, appearance, color, look, finish, or exterior is pretty/good/satisfying",
            "weight is heavy, heavier than expected, burdensome, tiring to hold, or disappointing because of weight"
        ],
        "exclude": [
            "only design praise",
            "only weight complaint without design-positive contrast"
        ],
    },
    "Q13": {
        "intent_type": "contrast_rating",
        "target_sentiment": "mixed",
        "must_match": [
            "product itself, phone, item, performance, or function is good/fine/satisfactory",
            "bad packaging, no cushioning, damaged/crushed/dented box, torn package, poor delivery packaging, or packaging defect",
            "low star rating, rating lowered, gave fewer stars, or says stars were deducted because of packaging"
        ],
        "exclude": [
            "bad product itself",
            "packaging issue without rating lowered/product-good contrast"
        ],
    },
    "Q14": {
        "intent_type": "positive_topic",
        "target_sentiment": "positive",
        "must_match": [
            "screen protection, privacy protection, privacy filter, anti-peeping, screen protector, tempered glass, film, or privacy screen feature",
            "satisfaction with protection/privacy function, visibility control, screen guarding, or protective film quality"
        ],
        "exclude": [
            "privacy/protection complaint",
            "generic screen satisfaction without protection/privacy/filter/film context"
        ],
    },
    "Q15": {
        "intent_type": "contrast",
        "target_sentiment": "mixed",
        "must_match": [
            "camera, photo, picture, video, or basic camera quality is good/satisfying",
            "telephoto, zoom, zoom-in, magnification, distant subject, or close-up zoom is disappointing, blurry, poor, lacking, or not as good"
        ],
        "exclude": [
            "only camera praise",
            "only zoom complaint without overall camera-positive contrast"
        ],
    },
}


def load_data():
    with REVIEWS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        str(r.get("reviewId")): r.get("content", "")
        for r in data
        if r.get("reviewId") and r.get("content")
    }


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"queries": {}}


def save_json(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    last_error = None
    for attempt in range(5):
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))

    backup_path = path.with_suffix(path.suffix + ".backup")
    with backup_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Warning: could not replace {path}; wrote backup to {backup_path}. Last error: {last_error}")


def parse_llm_response(raw_text):
    if not raw_text:
        return {}

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(raw_text)

    for text in candidates:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


def infer_query_intent(query):
    intent = QUERY_INTENTS.get(query["query_id"], {}).copy()
    if intent:
        return intent
    return {
        "intent_type": "semantic",
        "target_sentiment": "any",
        "must_match": [query["query"]],
        "exclude": ["unrelated reviews"],
    }


def normalize_model_results(parsed_data, id_map, query_id):
    clean = {}
    for rid, value in parsed_data.items():
        rid = str(rid)
        if rid not in id_map or not isinstance(value, dict):
            continue

        try:
            score = int(value.get("score", 0))
        except (TypeError, ValueError):
            score = 0

        evidence = str(value.get("evidence", "")).strip()
        reasoning = str(value.get("reasoning", "")).strip()
        matched_conditions = value.get("matched_conditions", [])
        if not isinstance(matched_conditions, list):
            matched_conditions = []

        if score < MIN_SCORE:
            continue
        if not evidence or evidence not in id_map[rid]:
            continue
        if not passes_query_specific_filter(query_id, id_map[rid], value):
            continue

        clean[rid] = {
            "score": score,
            "sentiment": str(value.get("sentiment", "")).strip().lower(),
            "intent_type": str(value.get("intent_type", "")).strip(),
            "matched_conditions": matched_conditions,
            "evidence": evidence,
            "reasoning": reasoning,
        }
    return clean


def passes_query_specific_filter(query_id, review_text, value):
    evidence = str(value.get("evidence", "")).strip()
    reasoning = str(value.get("reasoning", "")).strip()
    combined = f"{review_text} {evidence} {reasoning}"

    if query_id == "Q02":
        white_terms = ["화이트", "흰색", "하얀", "하얗", "white"]
        negative_color_terms = [
            "불만", "별로", "아쉽", "실망", "후회", "누렇", "노랗", "변색",
            "때", "오염", "색이", "색상", "색감", "기대", "생각보다", "실물",
            "구하기 힘들", "라벤더", "재입고", "그냥 화이트"
        ]
        positive_only_terms = [
            "깔끔", "예뻐", "예쁘", "영롱", "고급", "만족", "잘 했", "후회는 없"
        ]

        has_white = any(term.lower() in combined.lower() for term in white_terms)
        has_negative_color = any(term in combined for term in negative_color_terms)
        evidence_sounds_positive = any(term in evidence for term in positive_only_terms)
        if not has_white or not has_negative_color or evidence_sounds_positive:
            return False

    return True


def build_final_output(checkpoint, id_map):
    final_output = {}
    for query_id, q_state in checkpoint.get("queries", {}).items():
        per_model_top10 = {}
        for model_name, model_results in q_state.get("results", {}).items():
            top_10 = sorted(
                model_results.items(),
                key=lambda x: x[1].get("score", 0),
                reverse=True,
            )[:20]
            per_model_top10[model_name] = [
                {"rank": i + 1, "reviewId": rid, **val, "content": id_map.get(rid, "")}
                for i, (rid, val) in enumerate(top_10)
            ]

        final_output[query_id] = {
            "query": q_state.get("query", ""),
            "category": q_state.get("category", "Unknown"),
            "intent": q_state.get("intent", {}),
            "completed_chunks": q_state.get("completed_chunks", []),
            "completed_chunks_by_model": q_state.get("completed_chunks_by_model", {}),
            "per_model": per_model_top10,
        }
    return final_output


async def call_api(session, url, headers, payload, api_type):
    async with SEM:
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=TIMEOUT) as resp:
                    body = await resp.text()
                    if resp.status == 429 and attempt < MAX_API_RETRIES:
                        retry_after = resp.headers.get("retry-after")
                        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else RATE_LIMIT_RETRY_SECONDS
                        print(f"      {api_type} rate limited; waiting {wait_seconds}s before retry {attempt + 1}/{MAX_API_RETRIES}")
                        await asyncio.sleep(wait_seconds)
                        continue

                    if resp.status != 200:
                        print(f"      {api_type} API error {resp.status}: {body[:200]}")
                        return None

                    res = json.loads(body)
                    if api_type == "openai":
                        text = res["choices"][0]["message"]["content"]
                    elif api_type == "claude":
                        text = res["content"][0]["text"]
                    elif api_type == "gemini":
                        text = res["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        return None

                    parsed = parse_llm_response(text)
                    if not parsed:
                        print(f"      {api_type} returned no parseable JSON")
                        return None
                    return parsed
            except Exception as exc:
                if attempt < MAX_API_RETRIES:
                    print(f"      {api_type} failed: {type(exc).__name__}; retrying in 10s")
                    await asyncio.sleep(10)
                    continue
                print(f"      {api_type} failed: {type(exc).__name__}: {exc}")
                return None
        return None


def build_prompt(query, intent, chunk_ids, id_map):
    chunk_str = "\n\n".join(
        f"ID: {rid}\nCONTENT: {id_map[rid]}"
        for rid in chunk_ids
    )

    return f"""Search query: {query["query"]}
Intent type: {intent["intent_type"]}
Target sentiment: {intent["target_sentiment"]}
Must match all of these conditions:
{json.dumps(intent["must_match"], ensure_ascii=False, indent=2)}
Exclude reviews matching any of these conditions:
{json.dumps(intent["exclude"], ensure_ascii=False, indent=2)}

Task:
Select only reviews that directly satisfy the query intent. Some queries are about sentiment, some are about purchase context, comparison, rating contrast, or mixed positive/negative conditions. Judge the full intent, not just keywords.

Rules:
1. Use only facts explicitly written in CONTENT.
2. Do not invent missing evidence or reasons.
3. evidence must be a short exact substring copied from CONTENT.
4. If evidence is not literally present in CONTENT, do not output that review.
5. Do not force 10 results. Output only strong matches.
6. Include only reviews with score >= 60.
7. The sentiment field must be one of: positive, negative, mixed, neutral.
8. For mixed/contrast queries, the review must satisfy both sides of the contrast.
9. Output only one JSON object. No markdown.

Output format:
{{
  "reviewId": {{
    "score": 90,
    "sentiment": "positive|negative|mixed|neutral",
    "intent_type": "{intent["intent_type"]}",
    "matched_conditions": ["which required conditions were satisfied"],
    "evidence": "exact substring copied from CONTENT",
    "reasoning": "one short Korean sentence explaining why it matches the full query intent"
  }}
}}

[Reviews]
{chunk_str}
"""


async def main():
    id_map = load_data()
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    # Test mode: run only Q01. Comment this line out to run all 15 queries.
    # queries = queries[:1]

    checkpoint = load_checkpoint()
    model_names = MODEL_NAMES
    print(f"Running models: {', '.join(model_names)}")

    ids = list(id_map.keys())
    total_chunks = (len(ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for q in queries:
            query_id = q["query_id"]
            intent = infer_query_intent(q)
            print(f"\nScoring [{query_id}] {q['query']} ({intent['intent_type']})")

            q_state = checkpoint["queries"].setdefault(query_id, {
                "query": q["query"],
                "category": q.get("category", "Unknown"),
                "intent": intent,
                "completed_chunks": [],
                "completed_chunks_by_model": {m: [] for m in model_names},
                "results": {m: {} for m in model_names},
            })
            q_state["query"] = q["query"]
            q_state["category"] = q.get("category", "Unknown")
            q_state["intent"] = intent
            q_state.setdefault("completed_chunks", [])
            had_model_chunk_state = "completed_chunks_by_model" in q_state
            q_state.setdefault("completed_chunks_by_model", {})
            q_state.setdefault("results", {})
            if not had_model_chunk_state and q_state["completed_chunks"]:
                for model_name in model_names:
                    q_state["completed_chunks_by_model"][model_name] = list(q_state["completed_chunks"])
            for model_name in model_names:
                q_state["completed_chunks_by_model"].setdefault(model_name, [])
                q_state["results"].setdefault(model_name, {})

            completed_chunks_by_model = {
                model_name: set(q_state["completed_chunks_by_model"].get(model_name, []))
                for model_name in model_names
            }

            for start in range(0, len(ids), CHUNK_SIZE):
                chunk_no = start // CHUNK_SIZE + 1
                chunk_key = str(start)
                pending_models = [
                    model_name
                    for model_name in model_names
                    if chunk_key not in completed_chunks_by_model[model_name]
                ]
                if not pending_models:
                    print(f"   -> chunk {chunk_no}/{total_chunks} already done, skipping")
                    continue

                print(f"   -> chunk {chunk_no}/{total_chunks} processing ({', '.join(pending_models)})")
                chunk_ids = ids[start:start + CHUNK_SIZE]
                prompt = build_prompt(q, intent, chunk_ids, id_map)

                task_specs = []
                if "openai" in pending_models:
                    task_specs.append((
                        "openai",
                        call_api(
                            session,
                            "https://api.openai.com/v1/chat/completions",
                            {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                            {
                                "model": "gpt-4o-mini",
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0,
                                "response_format": {"type": "json_object"},
                            },
                            "openai",
                        ),
                    ))

                if "claude" in pending_models:
                    task_specs.append((
                        "claude",
                        call_api(
                            session,
                            "https://api.anthropic.com/v1/messages",
                            {
                                "x-api-key": os.getenv("CLAUDE_API_KEY"),
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json",
                            },
                            {
                                "model": CLAUDE_MODEL,
                                "max_tokens": 4096,
                                "messages": [{"role": "user", "content": prompt}],
                            },
                            "claude",
                        ),
                    ))

                if "gemini" in pending_models:
                    task_specs.append((
                        "gemini",
                        call_api(
                            session,
                            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={os.getenv('GEMINI_API_KEY')}",
                            {},
                            {"contents": [{"parts": [{"text": prompt}]}]},
                            "gemini",
                        ),
                    ))

                results = await asyncio.gather(*(task for _, task in task_specs))
                failed = False
                for (model_name, _), parsed_data in zip(task_specs, results):
                    if parsed_data is None:
                        failed = True
                        print(f"      {model_name}: failed; this chunk will retry next run")
                        continue

                    clean_results = normalize_model_results(parsed_data, id_map, query_id)
                    print(f"      {model_name}: kept {len(clean_results)} grounded matches")
                    for rid, data in clean_results.items():
                        old_score = q_state["results"][model_name].get(rid, {}).get("score", -1)
                        if data["score"] > old_score:
                            q_state["results"][model_name][rid] = data
                    q_state["completed_chunks_by_model"][model_name].append(chunk_key)
                    completed_chunks_by_model[model_name].add(chunk_key)

                save_json(CHECKPOINT_PATH, checkpoint)
                save_json(OUTPUT_PATH, build_final_output(checkpoint, id_map))
                if failed:
                    print(f"   -> chunk {chunk_no}/{total_chunks} saved partial results; will retry next run")
                else:
                    if all(chunk_key in completed_chunks_by_model[model_name] for model_name in model_names):
                        q_state["completed_chunks"].append(chunk_key)
                    save_json(CHECKPOINT_PATH, checkpoint)
                    save_json(OUTPUT_PATH, build_final_output(checkpoint, id_map))
                    print(f"   -> chunk {chunk_no}/{total_chunks} saved")

                if chunk_no < total_chunks:
                    wait_seconds = CHUNK_DELAY_SECONDS if "claude" not in pending_models else max(CHUNK_DELAY_SECONDS, 65)
                    print(f"   -> waiting {wait_seconds}s to stay under rate limit")
                    await asyncio.sleep(wait_seconds)

    save_json(CHECKPOINT_PATH, checkpoint)
    save_json(OUTPUT_PATH, build_final_output(checkpoint, id_map))
    print("\nDone. Results saved.")


if __name__ == "__main__":
    asyncio.run(main())
