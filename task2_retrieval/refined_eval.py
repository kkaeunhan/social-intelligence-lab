import asyncio
import json
import os
import re
import time
from pathlib import Path
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# API 부하 및 Rate Limit 방지를 위한 동시성 제어
CHUNK_SIZE = int(os.getenv("OPENAI_CHUNK_SIZE", "25"))
MIN_SCORE = 70
PROMPT_VERSION = "strict_query_conditions_v10_q06_q12_strict"
TOP_K = 20
PARALLEL_CHUNKS = int(os.getenv("OPENAI_PARALLEL_CHUNKS", "3"))
SEM = asyncio.Semaphore(PARALLEL_CHUNKS)
TIMEOUT = aiohttp.ClientTimeout(total=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "120")))
MAX_API_RETRIES = 4
RATE_LIMIT_RETRY_SECONDS = int(os.getenv("RATE_LIMIT_RETRY_SECONDS", "70"))

# 경로 설정
QUERIES_PATH = Path("task2_retrieval/queries_final.json")
REVIEWS_PATH = Path("output/task2_retrieval/indexed_documents.jsonl") # 전처리 데이터 경로
CHECKPOINT_PATH = Path("task2_retrieval/final_evaluation_checkpoint_v3.json")
OUTPUT_PATH = Path("task2_retrieval/eval/openai_eval_v2.json")

MODEL_NAMES = ["openai"]
DEFAULT_RERUN_QUERY_IDS = ["Q06", "Q12"]
RERUN_QUERY_IDS = [
    query_id.strip()
    for query_id in os.getenv("RERUN_QUERY_IDS", ",".join(DEFAULT_RERUN_QUERY_IDS)).split(",")
    if query_id.strip()
]

QUERY_CONDITIONS = {
    "Q01": {
        "must_match": [
            "뽁뽁이, 뾱뾱이, 완충재, 에어캡 등 보호 포장 언급",
            "포장 상태에 대한 만족 또는 긍정",
        ],
        "exclude": ["뽁뽁이/완충재가 없다는 불만", "포장 불량, 박스 파손, 찌그러짐에 대한 불만"],
    },
    "Q02": {
        "must_match": ["제품이 아이폰이어야 함", "아이폰 또는 제품 무게 언급", "가볍다는 평가", "무게에 대한 만족 또는 긍정"],
        "exclude": ["갤럭시/삼성 제품 리뷰", "무겁다는 불만", "무게와 무관한 디자인/성능 만족"],
    },
    "Q03": {
        "must_match": ["자식, 자녀, 아들, 딸, 아이 등 자녀 대상 언급", "그 자녀에게 사줬거나 선물했다는 구매 맥락"],
        "exclude": ["부모님, 엄마, 아빠, 배우자, 연인, 친구, 지인에게 사준 리뷰", "아이/자녀가 단순히 이야기 배경으로만 언급됨", "본인 사용 목적으로만 구매"],
    },
    "Q04": {
        "must_match": ["쿠팡 고객센터, 교환, 반품, 환불, 회수, 상담 중 하나 이상 실제 진행/시도/문의 경험", "그 과정에 대한 명확한 불만 또는 부정 경험"],
        "exclude": ["교환/반품을 고민만 한 리뷰", "교환/반품이 원활했다는 긍정", "제품 자체 불만만 있고 고객센터/교환/반품 맥락 없음"],
    },
    "Q05": {
        "must_match": ["가족(부모, 자식, 형제, 조부모, 배우자) 또는 지인(친구, 연인, 동료) 대상 언급", "그 대상에게 선물용으로 구매했거나 사줬다는 맥락"],
        "exclude": ["본인 사용 목적으로만 구매", "추천 표현만 있고 선물 맥락 없음"],
    },
    "Q06": {
        "must_match": ["며칠/몇 주/한 달/장기간 등 실제 사용 기간 또는 실사용 경험", "사용 후 드러난 배터리 단점 또는 발열 단점", "배터리/발열에 대한 명확한 부정 표현: 아쉽다, 불편하다, 문제, 광탈, 빨리 닳음, 뜨겁다, 열감, 짧다"],
        "exclude": ["배터리/발열이 좋아졌다는 리뷰", "배터리가 오래가거나 충분하다는 리뷰", "발열이 없거나 줄었다는 리뷰", "초기 데이터 이동 중 일시적 발열만 언급"],
    },
    "Q07": {
        "must_match": ["가격, 가성비, 가격 대비, 비싸다 중 하나 이상 언급", "성능/품질/기능/구성 대비 아쉽다는 부정 평가", "가격과 성능/품질/기능을 비교하는 문맥"],
        "exclude": ["가성비가 좋다는 긍정", "가격만 비싸다고 하고 성능/품질/기능 대비 불만 없음", "비싸지만 만족한다는 리뷰"],
    },
    "Q08": {
        "must_match": ["전작, 이전 모델, 구형, 전에 쓰던 기기 등 비교 대상 언급", "크게 바뀐 점이 없거나 변화가 적다는 평가"],
        "exclude": ["업그레이드가 크다는 긍정", "전작보다 확실히 좋아졌다는 평가", "전작 비교 없이 일반 평가만 있음"],
    },
    "Q09": {
        "must_match": ["고용량, 256GB, 512GB, 1TB 등 용량 선택 언급", "사진이나 영상 촬영/저장 때문에 선택했다는 이유"],
        "exclude": ["용량만 언급하고 사진/영상 이유 없음", "사진/영상만 언급하고 고용량 선택 이유 없음", "가격/할인 때문에 고용량을 선택한 리뷰"],
    },
    "Q10": {
        "must_match": ["디스플레이, 화면, 선명함, 밝기, 부드러움, 주사율 중 하나 이상 언급", "그 화면 품질에 대한 큰 만족 또는 긍정"],
        "exclude": ["화면 불만", "제품 전체 만족만 있고 화면 근거 없음", "카메라/성능 만족만 있고 화면 언급 없음"],
    },
    "Q11": {
        "must_match": ["RATING이 4 이하", "제품 자체는 좋거나 만족스럽다는 평가", "낮은 별점 또는 별점을 낮춘/감점한 명확한 이유"],
        "exclude": ["RATING이 5인 리뷰", "제품 자체도 부정적임", "낮은 별점 맥락 없이 제품 긍정만 있음"],
    },
    "Q12": {
        "must_match": ["디자인이 예쁘거나 마음에 든다는 긍정", "무게가 무겁거나 묵직하거나 손목에 부담된다는 부정"],
        "exclude": ["디자인 긍정만 있음", "무게 불만만 있고 디자인 긍정 없음", "무게가 가볍거나 부담 없다는 긍정 리뷰", "무게가 적당하거나 밸런스가 좋다는 리뷰"],
    },
    "Q13": {
        "must_match": ["RATING이 4 이하", "제품이 완전히 불량은 아니거나 사용 가능함", "포장 불량, 완충재 부족, 박스 찌그러짐/구겨짐/파손 중 하나 이상"],
        "exclude": ["RATING이 5인 리뷰", "제품 자체 불량만 문제인 리뷰", "박스가 멀쩡하거나 포장이 괜찮다는 리뷰"],
    },
    "Q14": {
        "must_match": ["프라이버시, 사생활 보호, 엿보기 방지, 옆사람 화면 가림, 보안 화면 중 하나 이상 언급", "그 프라이버시/사생활 보호 기능에 대한 만족"],
        "exclude": ["일반 보호필름/액정 보호만 있고 프라이버시나 사생활 보호 맥락 없음", "일반 화면 만족만 있고 보호 기능 맥락 없음", "프라이버시 기능 불만"],
    },
    "Q15": {
        "must_match": ["기본 카메라 또는 사진/영상 품질은 좋다는 긍정", "망원, 줌, 줌인, 확대 촬영은 아쉽다는 부정"],
        "exclude": ["카메라 긍정만 있음", "줌 불만만 있고 기본 카메라 긍정 없음", "줌/망원도 만족하거나 개선됐다는 리뷰", "카메라 자체가 전반적으로 나쁘다는 리뷰"],
    },
}

# 1. 전처리된 JSONL 데이터 로드 (모든 종류의 ID 키 호환 및 토큰 절약)
def load_preprocessed_jsonl():
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"❌ 에러: {REVIEWS_PATH} 경로에 전처리 파일이 없습니다.")
        
    id_map = {}
    print("📦 indexed_documents.jsonl 파일 읽기 시작 (dense 필드 제외 중)...")
    with REVIEWS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            
            # 🟢 [토큰 절약] 대용량 벡터/텍스트 필드를 메모리 적재 전 즉시 삭제
            doc.pop("dense_text", None)
            doc.pop("dense_vector", None)
            
            # 🟢 [핵심 수정] 어떤 형태의 ID 키(review_id, reviewId, id, _id)든 매핑되도록 강화
            rid = str(doc.get("review_id") or doc.get("reviewId") or doc.get("id") or doc.get("_id") or "")
            content = doc.get("content") or doc.get("text") or doc.get("review_content") or ""
            
            if rid and content:
                product_name = doc.get("product_name") or ""
                item_name = doc.get("item_name") or ""
                title = doc.get("title") or ""
                rating = doc.get("rating")
                id_map[rid] = {
                    "product_name": product_name,
                    "item_name": item_name,
                    "title": title,
                    "rating": rating,
                    "content": content,
                    "search_text": "\n".join(
                        part for part in [
                            f"상품명: {product_name}" if product_name else "",
                            f"옵션명: {item_name}" if item_name else "",
                            f"제목: {title}" if title else "",
                            f"RATING: {rating}" if rating is not None else "",
                            f"리뷰: {content}",
                        ] if part
                    ),
                }
                
    print(f"✅ 전처리 완료된 데이터 {len(id_map)}개 로드 성공!")
    if len(id_map) == 0:
        print("⚠️ 경고: 로드된 데이터가 0개입니다. indexed_documents.jsonl의 키 이름을 다시 확인하세요.")
    return id_map


def review_content(id_map, rid):
    value = id_map.get(rid, {})
    return value.get("content", "") if isinstance(value, dict) else value


def review_search_text(id_map, rid):
    value = id_map.get(rid, {})
    return value.get("search_text", "") if isinstance(value, dict) else value


def review_product_name(id_map, rid):
    value = id_map.get(rid, {})
    return value.get("product_name", "") if isinstance(value, dict) else ""


def review_item_name(id_map, rid):
    value = id_map.get(rid, {})
    return value.get("item_name", "") if isinstance(value, dict) else ""


def review_rating(id_map, rid):
    value = id_map.get(rid, {})
    return value.get("rating") if isinstance(value, dict) else None

# 2. 체크포인트 시스템 관련 함수들
def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_states": {}}

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        backup_path = path.with_suffix(path.suffix + ".backup")
        with backup_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# 3. LLM 응답 파서
def parse_llm_response(raw_text):
    if not raw_text: return {}
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not match: return {}
    try:
        return json.loads(match.group(0))
    except:
        return {}

# 4. OpenAI 호출 핵심부 (json_object 포맷 활성화로 안정성 업그레이드)
async def call_openai_judge(session, prompt):
    async with SEM:
        for attempt in range(MAX_API_RETRIES + 1):
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"} # 🟢 확실한 JSON 반환 강제
            }
            try:
                async with session.post(url, headers=headers, json=payload, timeout=TIMEOUT) as resp:
                    if resp.status == 429 and attempt < MAX_API_RETRIES:
                        retry_after = resp.headers.get("retry-after")
                        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else RATE_LIMIT_RETRY_SECONDS
                        wait_seconds = max(wait_seconds, 10 * (attempt + 1))
                        print(f"\n      OpenAI rate limit; {wait_seconds}s 후 재시도합니다.")
                        await asyncio.sleep(wait_seconds)
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"\n      OpenAI API error {resp.status}: {body[:200]}")
                        return None
                    res = await resp.json()
                    text = res["choices"][0]["message"]["content"]
                    return parse_llm_response(text)
            except Exception as exc:
                if attempt < MAX_API_RETRIES:
                    print(f"\n      OpenAI 호출 실패({type(exc).__name__}); 10초 후 재시도합니다.")
                    await asyncio.sleep(10)
                    continue
                print(f"\n      OpenAI 호출 실패({type(exc).__name__}): {exc}")
                return None
        return None


def build_prompt(q_id, query_str, chunk_ids, id_map):
    conditions = QUERY_CONDITIONS.get(q_id, {"must_match": [query_str], "exclude": ["검색어 조건과 반대인 리뷰"]})
    chunk_str = "\n\n".join([f"ID: {rid}\n{review_search_text(id_map, rid)}" for rid in chunk_ids])
    # 🟢 [프롬프트 가이드 보완] 명시된 검색 조건과 리뷰 본문 근거만으로 채점
    return f"""
    사용자 검색어: '{query_str}'

    [현재 쿼리의 필수 조건]
    {json.dumps(conditions["must_match"], ensure_ascii=False, indent=2)}

    [제외해야 할 리뷰]
    {json.dumps(conditions["exclude"], ensure_ascii=False, indent=2)}

    [분석할 리뷰 데이터셋]
    {chunk_str}

    [작업 지시사항]
    1. 오직 제공된 [분석할 리뷰 데이터셋]의 리뷰 내용에 직접 드러난 근거만 사용하세요.
    2. [현재 쿼리의 필수 조건]을 모두 동시에 만족하는 리뷰만 선택하세요. 조건 중 하나라도 빠지면 0점입니다.
    3. [제외해야 할 리뷰]에 해당하면 관련 단어가 있어도 반드시 0점입니다.
    4. 검색어에 긍정 조건이 있으면 부정/불만 리뷰는 0점이고, 검색어에 부정 조건이 있으면 긍정 리뷰는 0점입니다.
    5. 제품 자체 만족, 빠른 배송, 정상 작동, 하자 없음처럼 검색어 조건과 직접 관련 없는 내용은 근거로 보지 마세요.
    6. 리뷰 본문에 근거가 없거나 추론이 필요한 경우, 관련 주제가 일부 있어도 0점을 부여하세요.
    7. 점수는 반드시 100, 90, 80, 70, 0 중 하나만 부여하세요.
    8. 점수 기준:
       - 100점: 모든 핵심 조건이 리뷰 본문/상품명에 직접적이고 강하게 드러남
       - 90점: 모든 핵심 조건을 만족하지만 일부 표현이 100점보다 덜 강함
       - 80점: 모든 핵심 조건은 있으나 일부 근거가 간접적이거나 약함
       - 70점: 모든 핵심 조건은 대체로 맞지만 수동 검수가 필요한 약한 후보
       - 0점: 핵심 조건 중 하나라도 빠졌거나, 검색어 조건과 반대이거나, 추론 없이는 판단 불가함
    9. 0점 리뷰는 출력하지 마세요. 70점 이상 리뷰만 출력하고, 적합한 리뷰가 없으면 빈 JSON 객체 {{}}만 출력하세요.
    10. 각 리뷰별로 왜 이 점수인지 'reasoning'을 1문장으로 간결하게 작성하세요. reasoning에는 만족한 핵심 조건을 구체적으로 언급하세요.
    11. 아래의 JSON 형식으로만 완벽하게 출력하세요. 키(Key) 이름은 반드시 제공된 데이터셋의 실제 리뷰 ID 번호여야 합니다.
    
    {{
      "실제리뷰ID번호": {{"score": 100 또는 90 또는 80 또는 70 또는 0, "reasoning": "점수 판단 이유"}}
    }}
    """


async def score_chunk(session, q_id, query_str, chunk_ids, id_map):
    prompt = build_prompt(q_id, query_str, chunk_ids, id_map)
    return await call_openai_judge(session, prompt)


def passes_query_filter(q_id, rid, score_info, id_map):
    text = review_search_text(id_map, rid)
    compact = text.replace(" ", "")
    rating = review_rating(id_map, rid)

    def has_any(terms):
        return any(term in text for term in terms)

    def has_any_compact(terms):
        return any(term.replace(" ", "") in compact for term in terms)

    def lacks_all(terms):
        return not has_any(terms)

    wrap_terms = ["뽁뽁", "뾱뾱", "완충재", "완충 포장", "에어캡", "보호포장", "보호 포장"]
    positive_terms = ["만족", "좋", "잘", "안전", "꼼꼼", "야무지", "가득", "합격", "손상 없이", "이상 없이", "감싸", "보호", "멀쩡"]
    negative_terms = [
        "뽁뽁이 없이", "뾱뾱이 없이", "완충재 없이", "포장 없이", "하나 없이",
        "뽁뽁이도 없이", "뾱뾱이도 없이", "완충재도 없이", "안 처넣", "덜렁",
        "허접", "짜증", "불만", "기분좋지", "기분 좋지",
        "뽁뽁이가 충분히", "충분히 쌓여있는 그런 형태가 아니라",
        "달랑", "박스도 다 찌그러", "택배 보고 이거 설마"
    ]

    if q_id == "Q01":
        return has_any(wrap_terms) and has_any(positive_terms) and lacks_all(negative_terms)

    if q_id == "Q02":
        iphone_terms = ["아이폰", "iPhone", "IPHONE", "Apple", "애플"]
        galaxy_terms = ["갤럭시", "Galaxy", "GALAXY", "삼성", "Samsung", "SAMSUNG"]
        weight_terms = ["무게", "가볍", "가벼", "가뿐", "경량", "한손", "한 손"]
        heavy_terms = ["무겁", "묵직", "아쉽"]
        return has_any(iphone_terms) and lacks_all(galaxy_terms) and has_any(weight_terms) and has_any(["가볍", "가벼", "가뿐", "경량"]) and lacks_all(heavy_terms)

    if q_id == "Q03":
        child_terms = ["자식", "자녀", "아들", "딸", "아이", "애들", "아들내미", "딸아이", "초등", "중등", "고등", "학생"]
        buy_terms = ["사줬", "사 주", "사주", "구매해줬", "구매해 줬", "선물", "입학", "졸업"]
        non_child_recipients = ["부모님", "부모", "엄마", "아빠", "어머니", "아버지", "남편", "아내", "와이프", "여자친구", "남자친구", "친구", "지인", "연인", "애인"]
        return has_any(child_terms) and has_any(buy_terms) and lacks_all(non_child_recipients)

    if q_id == "Q04":
        service_terms = ["고객센터", "상담", "교환", "반품", "환불", "회수", "쿠팡", "판매자", "AS", "A/S"]
        action_terms = ["했", "신청", "문의", "요청", "접수", "연락", "전화", "처리", "받", "거절", "지연", "안됨", "안 해", "안해"]
        complaint_terms = ["불만", "짜증", "화나", "최악", "실망", "거절", "지연", "늦", "안됨", "안 해", "안해", "어렵", "번거", "문제", "불편"]
        vague_terms = ["고민", "할까", "할지", "생각 중", "생각중", "해야 하나", "해야하나"]
        return has_any(service_terms) and has_any(action_terms) and has_any(complaint_terms) and lacks_all(vague_terms)

    if q_id == "Q05":
        recipient_terms = [
            "가족", "지인", "부모", "부모님", "엄마", "아빠", "어머니", "아버지",
            "할머니", "할아버지", "조부모", "남편", "아내", "와이프", "배우자",
            "형", "누나", "언니", "오빠", "동생", "친구", "여자친구", "남자친구",
            "연인", "애인", "동료", "아이", "자녀", "아들", "딸"
        ]
        gift_terms = ["선물", "생일", "기념일", "사줬", "사 주", "사주", "드리", "드렸", "해드"]
        return has_any(recipient_terms) and has_any(gift_terms)

    if q_id == "Q06":
        use_period_terms = ["사용 후", "써보", "쓰다 보", "며칠", "몇 주", "몇주", "몇 달", "몇달", "한 달", "한달", "장기간", "실사용", "동안"]
        issue_terms = ["배터리", "발열", "뜨거", "뜨겁", "열감", "광탈", "빨리 닳", "빨리닳", "닳", "오래 못", "짧"]
        issue_negative_terms = ["아쉽", "단점", "문제", "불편", "광탈", "빨리", "뜨거", "뜨겁", "열감", "짧", "오래 못", "부족", "소모", "닳"]
        positive_only = [
            "배터리 만족", "배터리 오래", "발열 없", "발열이 없", "문제 없이",
            "오래가", "오래 가", "하루 종일", "충분", "우수", "뛰어나",
            "개선", "잡힌", "좋아졌", "좋고", "좋았", "안정적", "줄어"
        ]
        transient_terms = ["초기 세팅", "초기 설정", "데이터 이동", "마이그레이션"]
        return has_any(use_period_terms) and has_any(issue_terms) and has_any(issue_negative_terms) and lacks_all(positive_only) and lacks_all(transient_terms)

    if q_id == "Q07":
        price_terms = ["가격", "가성비", "가격 대비", "값", "비싸", "금액"]
        performance_terms = ["성능", "기능", "품질", "스펙", "사양", "구성", "배터리", "카메라", "화면", "용량"]
        value_negative_terms = ["아쉽", "별로", "불만", "비싸", "부족", "실망", "그 돈", "돈값", "대비", "과하", "아깝"]
        positive_value_terms = ["가성비 좋", "가격 대비 좋", "가격대비 좋", "만족", "괜찮", "납득", "메리트"]
        return has_any(price_terms) and has_any(performance_terms) and has_any(value_negative_terms) and lacks_all(positive_value_terms)

    if q_id == "Q08":
        previous_terms = ["전작", "이전", "전 모델", "이전 모델", "구형", "전에 쓰", "기존", "바뀐"]
        little_change_terms = ["차이 없", "차이가 없", "큰 차이", "크게 바뀐", "바뀐 점이 없", "혁신", "비슷", "똑같", "그대로"]
        positive_upgrade_terms = ["많이 좋아", "확실히 좋아", "체감이 크", "체감 큼", "업그레이드 만족", "완전 달라"]
        return has_any(previous_terms) and has_any(little_change_terms) and lacks_all(positive_upgrade_terms)

    if q_id == "Q09":
        capacity_terms = ["256GB", "512GB", "1TB", "256기가", "512기가", "1테라", "고용량", "용량", "저장공간", "저장 공간"]
        media_terms = ["사진", "영상", "동영상", "촬영", "카메라", "저장"]
        price_reason_terms = ["할인", "가격", "혜택", "업그레이드로", "무료 업그레이드"]
        return has_any(capacity_terms) and has_any(media_terms) and not (has_any(price_reason_terms) and lacks_all(["사진", "영상", "동영상", "촬영"]))

    if q_id == "Q10":
        display_terms = ["디스플레이", "화면", "선명", "부드러", "주사율", "120Hz", "프로모션", "스크롤", "밝기", "화질"]
        display_positive_terms = ["만족", "좋", "선명", "부드러", "쨍", "밝", "체감", "대박"]
        display_negative_terms = ["불만", "아쉽", "별로", "어둡", "잔상"]
        return has_any(display_terms) and has_any(display_positive_terms) and lacks_all(display_negative_terms)

    if q_id == "Q11":
        product_positive_terms = ["제품은 좋", "제품 자체는 좋", "제품 자체는 괜찮", "기기는 좋", "물건은 좋", "성능은 좋", "폰은 좋", "만족"]
        rating_reason_terms = ["배송", "포장", "박스", "고객센터", "교환", "반품", "오염", "찌그러", "늦"]
        return rating is not None and rating <= 4 and has_any(product_positive_terms) and has_any(rating_reason_terms)

    if q_id == "Q12":
        design_terms = ["디자인", "색상", "컬러", "외관", "예쁘", "이쁘", "고급", "깔끔"]
        heavy_terms = ["무겁", "무거", "묵직", "손목에 부담", "손목 부담", "부담"]
        light_positive_terms = [
            "가볍", "가벼", "부담이 덜", "부담 덜", "부담이 적", "부담 적",
            "부담 없", "부담없", "아쉬움이 없", "밸런스", "적당", "편하"
        ]
        return has_any(design_terms) and has_any(["예쁘", "이쁘", "고급", "깔끔", "만족", "좋"]) and has_any(heavy_terms) and lacks_all(light_positive_terms)

    if q_id == "Q13":
        product_ok_terms = ["제품은 좋", "제품 자체는 좋", "제품은 괜찮", "기기는 좋", "물건은 좋", "폰은 좋", "성능은 좋", "사용은", "이상 없", "정상", "멀쩡", "괜찮"]
        package_bad_terms = ["포장 불량", "박스", "찌그러", "구겨", "찢어", "파손", "완충재 없이", "뽁뽁이 없이", "덜렁", "오염", "훼손", "포장"]
        package_ok_terms = ["박스 구겨짐 없이", "박스가 멀쩡", "포장 잘", "안전하게"]
        return rating is not None and rating <= 4 and has_any(product_ok_terms) and has_any(package_bad_terms) and lacks_all(package_ok_terms)

    if q_id == "Q14":
        privacy_terms = ["프라이버시", "사생활", "프라이버시 필름", "가림", "엿보기", "보안", "옆사람", "옆에서", "옆에"]
        privacy_positive_terms = ["만족", "좋", "잘", "보호", "안심", "편", "유용"]
        return has_any(privacy_terms) and has_any(privacy_positive_terms)

    if q_id == "Q15":
        camera_positive_terms = ["카메라", "사진", "영상", "화질"]
        positive_terms_q15 = ["좋", "만족", "잘 나", "선명", "예쁘", "괜찮"]
        zoom_terms = ["망원", "줌", "줌인", "확대", "멀리"]
        zoom_negative_terms = ["아쉽", "부족", "흐릿", "별로", "깨짐", "화질 저하", "뭉개"]
        zoom_positive_terms = ["줌도 좋", "망원도 좋", "확대도 좋", "줌 기능이 뛰어나", "개선되", "유용", "선명"]
        camera_bad_terms = ["카메라가 최악", "카메라 최악", "카메라 별로", "카메라가 별로"]
        return has_any(camera_positive_terms) and has_any(positive_terms_q15) and has_any(zoom_terms) and has_any(zoom_negative_terms) and lacks_all(zoom_positive_terms) and lacks_all(camera_bad_terms)

    return True

# 5. 최종 리포트 파일 업데이트 빌더 (Top K 정밀 추출)
def update_final_report(checkpoint_state, id_map, queries):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        try:
            with OUTPUT_PATH.open("r", encoding="utf-8") as f:
                final_output = json.load(f)
        except Exception:
            final_output = {}
    else:
        final_output = {}
    
    for q in queries:
        q_id = q['query_id']
        if q_id in checkpoint_state["completed_states"]:
            q_state = checkpoint_state["completed_states"][q_id]
            all_scores = q_state.get("scores", {})
            
            # score 높은 순 정렬 후 Top K개만 추출
            top_k = sorted(all_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:TOP_K]
            
            final_output[q_id] = {
                "query": q['query'],
                "category": q.get('category', 'Unknown'),
                "reviews_to_evaluate": [
                    {
                        "rank": rank + 1,
                        "reviewId": rid,
                        "score": info["score"],
                        "reasoning": info["reasoning"],
                        "product_name": review_product_name(id_map, rid),
                        "item_name": review_item_name(id_map, rid),
                        "rating": review_rating(id_map, rid),
                        "content": review_content(id_map, rid)
                    }
                    for rank, (rid, info) in enumerate(top_k) if info["score"] >= MIN_SCORE
                ]
            }
            
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

async def main():
    # 🟢 [테스트 플래그] True면 Q01만 실행 / False면 전체 실행
    TEST_ONLY_Q01 = False 
    
    id_map = load_preprocessed_jsonl()
    if len(id_map) == 0:
        return # 데이터가 없으면 진행하지 않고 종료
        
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    if TEST_ONLY_Q01:
        print("\n🔬 [테스트 모드 활성화] 첫 번째 쿼리(Q01)에 대해서만 실험을 진행합니다.")
        queries = queries[:1]
    elif RERUN_QUERY_IDS:
        wanted = set(RERUN_QUERY_IDS)
        queries = [q for q in queries if q["query_id"] in wanted]
        print(f"\n🔁 선택 쿼리만 재실행합니다: {', '.join(q['query_id'] for q in queries)}")
        
    checkpoint = load_checkpoint()
    ids = list(id_map.keys())
    total_chunks = (len(ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for q in queries:
            q_id = q['query_id']
            query_str = q['query']
            
            # 기존 체크포인트가 완전히 밀려있거나 새로 시작할 때 초기화
            if (
                q_id not in checkpoint["completed_states"]
                or checkpoint["completed_states"][q_id].get("fully_completed", False)
                or checkpoint["completed_states"][q_id].get("chunk_size") != CHUNK_SIZE
                or checkpoint["completed_states"][q_id].get("prompt_version") != PROMPT_VERSION
            ):
                checkpoint["completed_states"][q_id] = {
                    "last_chunk_index": 0,
                    "scores": {},
                    "fully_completed": False,
                    "chunk_size": CHUNK_SIZE,
                    "prompt_version": PROMPT_VERSION,
                }
                
            q_state = checkpoint["completed_states"][q_id]
            print(f"\n🔥 채점 시작: [{q_id}] {query_str}")
            
            all_scores = q_state.get("scores", {})
            start_idx = q_state.get("last_chunk_index", 0)
            
            # 100개씩 나눈 청크를 여러 개 병렬 처리
            batch_size = CHUNK_SIZE * PARALLEL_CHUNKS
            for batch_start in range(start_idx, len(ids), batch_size):
                starts = list(range(batch_start, min(batch_start + batch_size, len(ids)), CHUNK_SIZE))
                first_chunk = (starts[0] // CHUNK_SIZE) + 1
                last_chunk = (starts[-1] // CHUNK_SIZE) + 1
                print(f"   -> 청크 {first_chunk}~{last_chunk}/{total_chunks} 병렬 분석 중...")

                tasks = [
                    score_chunk(session, q_id, query_str, ids[start:start + CHUNK_SIZE], id_map)
                    for start in starts
                ]
                chunk_results = await asyncio.gather(*tasks)
                had_failure = any(result is None for result in chunk_results)

                # 결과 취합
                for parsed_data in chunk_results:
                    if parsed_data is None:
                        continue
                    for rid, val in parsed_data.items():
                        rid = str(rid)
                        if rid in id_map and isinstance(val, dict):
                            try:
                                raw_score = max(0, min(100, int(val.get("score", 0))))
                            except:
                                raw_score = 0

                            if raw_score >= 95:
                                score = 100
                            elif raw_score >= 85:
                                score = 90
                            elif raw_score >= 75:
                                score = 80
                            elif raw_score >= 70:
                                score = 70
                            else:
                                score = 0
                            
                            if score >= MIN_SCORE:
                                score_info = {
                                    "score": score,
                                    "reasoning": val.get("reasoning", "추론 근거 누락")
                                }
                                if (
                                    passes_query_filter(q_id, rid, score_info, id_map)
                                    and score > all_scores.get(rid, {}).get("score", -1)
                                ):
                                    all_scores[rid] = {
                                        "score": score,
                                        "reasoning": val.get("reasoning", "추론 근거 누락")
                                    }
                
                # 배치 단위 체크포인트 저장. 실패가 있으면 같은 배치를 다음 실행에서 재시도합니다.
                if had_failure:
                    print("      일부 청크가 실패하여 checkpoint를 전진하지 않습니다. 다음 실행에서 이 배치를 재시도합니다.")
                else:
                    q_state["last_chunk_index"] = starts[-1] + CHUNK_SIZE
                q_state["scores"] = all_scores
                save_json(CHECKPOINT_PATH, checkpoint)
                update_final_report(checkpoint, id_map, queries)
                if had_failure:
                    return
                
            # 완료 마킹 및 임시 리포트 덤프
            q_state["fully_completed"] = True
            save_json(CHECKPOINT_PATH, checkpoint)
            update_final_report(checkpoint, id_map, queries)
            print(f"   -> 🎉 [{q_id}] 데이터셋 분석 및 Top {TOP_K} 파일 쓰기 완료!{' '*20}")

    print(f"\n🎉 모든 작업 완료! 최종 가공 파일 확인: {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
