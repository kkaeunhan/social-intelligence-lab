import json
import asyncio
import aiohttp
import re
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
SEM = asyncio.Semaphore(30)

def load_data():
    path = Path("si_dataset/review_for_analysis.json")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(r.get("reviewId")): r.get("content", "") for r in data if r.get("reviewId")}

def extract_scores(raw_text, id_map):
    # 정규식 수정: JSON 블록 내부만 보거나 전체에서 ID:점수 패턴 추출
    matches = re.findall(r'"(\d{8,})"\s*:\s*(\d{1,3})', raw_text)
    scores = {}
    for rid, score in matches:
        if rid in id_map:
            scores[rid] = int(score)
    return scores

async def call_api(session, url, headers, payload, api_type, id_map):
    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200: return {}
            res = await resp.json()
            if api_type == "openai": text = res["choices"][0]["message"]["content"]
            elif api_type == "claude": text = res["content"][0]["text"]
            elif api_type == "gemini": text = res['candidates'][0]['content']['parts'][0]['text']
            return extract_scores(text, id_map)
    except Exception as e:
        print(f"API Error ({api_type}): {e}")
        return {}

async def main():
    id_map = load_data()
    with open("task2_retrieval/queries_ex.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    ids = list(id_map.keys())
    final_output = {}

    async with aiohttp.ClientSession() as session:
        for q in queries:
            query_str = q['query']
            print(f"\n🔥 채점 중: [{q['query_id']}] {query_str}")
            
            all_scores = {} 
            
            # 20개씩 청크 처리
            for i in range(0, len(ids), 20):
                print(f"   -> 청크 {i//20 + 1}/{(len(ids)//20)+1} 처리 중...", end="\r")
                chunk_ids = ids[i:i+20]
                chunk_str = "\n".join([f"ID: {rid}, 내용: {id_map[rid]}" for rid in chunk_ids])
                
                prompt = f"""검색어: '{query_str}'
[데이터셋]
{chunk_str}

[지시사항]
1. 검색어와 의도가 일치하는 리뷰를 찾고 추론 과정을 작성하세요.
2. 관련성 점수(1~100)를 매겨 아래 JSON 형식으로 출력하세요.
{{"ID": 점수}}
"""
                async def safe_call(session, url, headers, payload, api_type, id_map):
                    async with SEM: 
                        return await call_api(session, url, headers, payload, api_type, id_map)
                
                # API 매칭
                tasks = [
                    call_api(session, "https://api.openai.com/v1/chat/completions", 
                             {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}, 
                             {"model": "gpt-4.1-mini", "messages": [{"role":"user", "content": prompt}], "temperature": 0}, 
                             "openai", id_map),
                    call_api(session, "https://api.anthropic.com/v1/messages", 
                             {"x-api-key": os.getenv("CLAUDE_API_KEY"), "anthropic-version": "2023-06-01", "content-type": "application/json"}, 
                             {"model": "claude-3-haiku-20240307", "max_tokens": 1000, "messages": [{"role":"user", "content": prompt}]}, 
                             "claude", id_map),
                    call_api(session, f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}", 
                             {}, {"contents": [{"parts": [{"text": prompt}]}]}, 
                             "gemini", id_map)
                ]
                
                results = await asyncio.gather(*tasks)
                for res_dict in results:
                    for rid, score in res_dict.items():
                        all_scores[rid] = max(all_scores.get(rid, 0), score)

            print()
            # 파이썬에서 즉시 정렬
            top_10_ids = sorted(all_scores, key=all_scores.get, reverse=True)[:10]
            
            final_output[q['query_id']] = {
                "query": query_str,
                "category": q.get('category', 'Unknown'),
                "reviews_to_evaluate": [{"rank": rank+1, "reviewId": rid, "score": all_scores[rid], "content": id_map[rid]} for rank, rid in enumerate(top_10_ids)]
            }

    with open("task2_retrieval/clean_evaluation_pool_ex.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 성공! JSON 결과 파일이 생성되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())