import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 에러: .env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return
        
    client = OpenAI(api_key=api_key)
    
    CURRENT_FILE_DIR = Path(__file__).resolve().parent 
    PROJECT_ROOT_DIR = CURRENT_FILE_DIR.parent          

    data_path = PROJECT_ROOT_DIR / "si_dataset" / "review_for_analysis.json"
    
    if not data_path.exists():
        print(f"❌ 에러: 원본 파일이 존재하지 않습니다. 경로 확인 요망: {data_path}")
        return

    print("🔄 원본 데이터 레이크 파일을 로드하고 있습니다...")
    with data_path.open(encoding="utf-8") as f:
        records = json.load(f)
        
    total_count = len(records)
    print(f"📋 전수조사 가동: 총 {total_count}개의 리뷰 전체 필드를 누락 없이 정독하는 Map-Reduce 파이프라인 가동!")

    batch_size = 100
    all_context_summaries = []
    
    for i in range(0, total_count, batch_size):
        chunk = records[i:i + batch_size]
        print(f"📦 [Map 단] 전 필드 무결성 정독 및 틈새 맥락 추출 중... ({i}/{total_count} 건 완료)")
        
        chunk_texts = ""
        for idx, r in enumerate(chunk):
            review_id = r.get("reviewId") or r.get("review_id") or "Unknown"
            p_name = r.get("product_name") or r.get("productName") or "Unknown"
            i_name = r.get("itemName") or r.get("item_name") or "None"
            title = r.get("title") or "제목 없음"
            content = r.get("content", "").strip()
            rating = r.get("rating", "N/A")
            review_at = r.get("reviewAt") or r.get("review_at") or "Unknown_Date"
            
            img_count = len(r.get("attachments") or [])
            vid_count = len(r.get("videoAttachments") or [])
            has_image = "Y" if img_count > 0 else "N"
            has_video = "Y" if vid_count > 0 else "N"
            
            helpful_count = r.get("helpful_count") or r.get("helpfulCount") or 0
            helpful_true_count = r.get("helpful_true_count") or r.get("helpfulTrueCount") or 0
            helpful_false_count = r.get("helpful_false_count") or r.get("helpfulFalseCount") or 0
            
            survey_answers = r.get("reviewSurveyAnswers") or r.get("review_survey_answers") or []
            survey_text = ""
            if survey_answers:
                survey_text = " | 설문답변: " + ", ".join([f"Q: {s.get('question')}->A: {s.get('answer')}" for s in survey_answers if s.get('question')])

            if not content:  
                continue
                
            chunk_texts += (
                f"[{idx}] 리뷰ID: {review_id} | 상품명: {p_name} ({i_name}) | 별점: {rating}점 | 작성일: {review_at}\n"
                f"▶ 추천수 관련 지표 -> 총추천: {helpful_count}회 | 유저공감(True): {helpful_true_count}회 | 유저비공감(False): {helpful_false_count}회\n"
                f"▶ 미디어 정황 -> 사진첨부: {has_image}({img_count}장) | 영상첨부: {has_video}({vid_count}개){survey_text}\n"
                f"▶ 텍스트 -> 제목: {title} | 본문: {content}\n"
                f"--------------------------------------------------\n"
            )
            
        map_prompt = f"""
        당신은 이커머스 전문 데이터 분석관입니다. 제공된 100개의 리뷰 원본 정보셋을 아주 깊게 정독하세요.
        단순 본문 뿐만 아니라 [별점], [유저공감(True) 횟수], [사진/영상 첨부여부] 등을 유기적으로 크로스 체크하여 향후 고차원 지능형 검색어의 훌륭한 재료가 될 만한 날카로운 마이크로 맥락 3~5가지를 한 문장씩 도출해 주세요.

        특히 다음 데이터 정황을 추적해야 합니다:
        1. 감성 모순 케이스: 본문 문장과 제목에서는 호평을 하는데 정작 [별점]은 1~2점으로 낮게 준 특이 데이터 정황
        2. 집단지성 신뢰 케이스: 특정 속성(예: 뽁뽁이 포장)에 만족했는데 일반 소비자들이 [유저공감(True)] 버튼을 압도적으로 많이 눌러 신뢰도가 입증된 진짜 클린 리뷰 맥락
        3. 정황 및 목적성 케이스: 자식 선물, 부모님 효도폰, 또는 색상 옵션 불만 등 구매 상황이 세밀하게 녹아든 데이터 정황

        [정독할 100건의 모든 필드 구조화 데이터]
        {chunk_texts}
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": map_prompt}],
            temperature=0.1
        )
        all_context_summaries.append(response.choices[0].message.content.strip())

    print("\n🚀 전수조사 완료: 추천수/공감수 정황까지 모두 반영된 보고서가 집적되었습니다. 15개 최종 쿼리를 빌드합니다...")
    combined_context = "\n===\n".join(all_context_summaries)
    
    # 🟢 1. 시스템 프롬프트: 2030 일반 사용자 페르소나 부여
    system_prompt = "너는 쿠팡/네이버쇼핑에서 스마트폰 리뷰를 검색하는 2030 일반 사용자야. 너무 딱딱한 전문 용어보다는 일상에서 대화하듯 편하고 쉬운 단어로 검색하는 편이야."
    
    # 🟢 2. 유저 프롬프트: "~한 리뷰" 형태 유지 + 쉬운 단어 강제 + 3분류
    user_prompt = f"""
    제공된 [데이터 맥락 요약본]을 바탕으로, 실제 일반 사용자가 검색창에 입력할 법한 자연스러운 검색 쿼리 15개를 생성해 줘.

    [핵심 작성 규칙 - 절대 엄수]
    1. 쿼리 형식: 모든 쿼리는 반드시 "~한 리뷰", "~인 리뷰" 형태 또는 "동생 사줄 최신 스마트폰"처럼 단어로 끝나야 해. (예: "뽁뽁이 포장 상태에 만족한 리뷰")
    2. 단어 선택 (제일 중요!): '주사율', '120Hz', '혁신' 같은 어려운 기술/전문 용어는 쓰는 걸 자제해줘. 대신 사용자가 일상적으로 느끼는 쉬운 표현으로 풀어서 써 줘.
       - 나쁜 예: "디스플레이 주사율에 만족한 리뷰" ➡️ 좋은 예: "화면 넘김이 부드러워서 만족한 리뷰"
       - 나쁜 예: "전작 대비 혁신이 없다는 리뷰" ➡️ 좋은 예: "이전 모델이랑 바뀐 게 없어서 아쉽다는 리뷰"
    
    [카테고리 구성 (각 5개씩 총 15개)]
    - "1_Short_Keyword": 짧고 단순한 상황 (예: "가족 선물용으로 구매한 리뷰", "화이트 색상 별로라는 리뷰")
    - "2_Medium_Context": 특정 경험이 들어간 중간 길이 (예: "한 달 썼는데 배터리가 금방 닳는 리뷰")
    - "3_Long_Complex": 복합적인 고민이나 딜레마 (예: "제품은 진짜 마음에 드는데 포장 찌그러져서 별점 낮게 준 리뷰")

    [데이터 맥락 요약본]
    {combined_context}

    [출력 형식]
    반드시 마크다운 외벽(```json) 없이 순수한 JSON 배열 형식만 반환해.
    [
      {{"query_id": "Q01", "query": "뽁뽁이 포장 상태에 만족한 리뷰", "category": "1_Short_Keyword"}},
      {{"query_id": "Q02", "query": "화면 넘김이 부드러워서 만족한 리뷰", "category": "2_Medium_Context"}},
      ... (총 15개)
    ]
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    output_text = response.choices[0].message.content.strip()
    
    try:
        if output_text.startswith("```json"):
            output_text = output_text.split("```json")[1].split("```")[0].strip()
        elif output_text.startswith("```"):
            output_text = output_text.split("```")[1].split("```")[0].strip()
            
        queries_json = json.loads(output_text)
        
        output_path = CURRENT_FILE_DIR / "queries.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(queries_json, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ 무결성 만점! 총 {len(queries_json)}개의 벤치마크 기출문제집 파일 생성이 완료되었습니다.")
        print(f"📂 파일 저장 성공 경로: {output_path}")
        
    except Exception as e:
        print(f"❌ 최종 JSON 파싱 결합 에러 발생: {e}")
        print(output_text)

if __name__ == "__main__":
    main()