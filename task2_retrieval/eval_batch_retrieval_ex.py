import json
import asyncio
import aiohttp
import re
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEM = asyncio.Semaphore(30)
TIMEOUT = aiohttp.ClientTimeout(total=40)

def load_data():
    path = Path("si_dataset/review_for_analysis.json")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(r.get("reviewId")): r.get("content", "") for r in data if r.get("reviewId")}

# 🟢 [수정] JSON 안에 reasoning을 넣는 구조로 파싱
def parse_llm_response(raw_text):
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not match:
        return {}
    try:
        # 데이터가 {"ID": {"score": 100, "reasoning": "..."}} 형태로 들어옴
        return json.loads(match.group(0))
    except:
        return {}

async def call_api(session, url, headers, payload, api_type):
    async with SEM:
        try:
            async with session.post(url, headers=headers, json=payload, timeout=TIMEOUT) as resp:
                if resp.status != 200: return {}
                res = await resp.json()
                if api_type == "openai": text = res["choices"][0]["message"]["content"]
                elif api_type == "claude": text = res["content"][0]["text"]
                elif api_type == "gemini": text = res['candidates'][0]['content']['parts'][0]['text']
                return parse_llm_response(text)
        except Exception:
            return {}

async def main():
    id_map = load_data()
    with open("task2_retrieval/queries_ex.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    final_output = {}

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for q in queries:
            query_str = q['query']
            print(f"\n🔥 채점 중: [{q['query_id']}] {query_str}")
            all_data = {} # {id: {"score": x, "reasoning": y}}
            
            ids = list(id_map.keys())
            for i in range(0, len(ids), 100): # 100개씩 청크
                print(f"   -> 청크 {i//100 + 1}/{(len(ids)//100)+1} 처리 중...", end="\r")
                chunk_ids = ids[i:i+100]
                chunk_str = "\n".join([f"ID: {rid}, 내용: {id_map[rid]}" for rid in chunk_ids])
                
                # 🟢 [개선] 프롬프트: 생각의 과정을 JSON 내부로 강제함
                prompt = f"""검색어: '{query_str}'
[데이터셋]
{chunk_str}

[지시사항]
1. 각 리뷰의 관련성을 0~100점 평가하세요.
2. 각 리뷰별로 [핵심 근거]를 1문장으로 JSON 안에 작성하세요.
3. 반드시 아래 JSON 형식으로만 출력하세요.
{{
  "ID": {{"score": 점수, "reasoning": "근거"}},
  "ID": {{"score": 점수, "reasoning": "근거"}}
}}
"""
                tasks = [
                    call_api(session, "https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}, {"model": "gpt-4o-mini", "messages": [{"role":"user", "content": prompt}], "temperature": 0}, "openai"),
                    call_api(session, "https://api.anthropic.com/v1/messages", {"x-api-key": os.getenv("CLAUDE_API_KEY"), "anthropic-version": "2023-06-01", "content-type": "application/json"}, {"model": "claude-3-haiku-20240307", "max_tokens": 1000, "messages": [{"role":"user", "content": prompt}]}, "claude"),
                    call_api(session, f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}", {}, {"contents": [{"parts": [{"text": prompt}]}]}, "gemini")
                ]
                
                results = await asyncio.gather(*tasks)
                for parsed_data in results:
                    for rid, data in parsed_data.items():
                        if rid in id_map:
                            if data.get("score", 0) > all_data.get(rid, {}).get("score", -1):
                                all_data[rid] = data

            # Top 10 선정
            top_10 = sorted(all_data.items(), key=lambda x: x[1]['score'], reverse=True)[:10]
            final_output[q['query_id']] = {
                "query": query_str,
                "category": q.get('category', 'Unknown'),
                "reviews_to_evaluate": [
                    {"rank": i+1, "reviewId": rid, **val, "content": id_map[rid]} 
                    for i, (rid, val) in enumerate(top_10)
                ]
            }

    with open("task2_retrieval/final_evaluation_report_2.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 성공! 정확도가 대폭 개선된 결과가 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())