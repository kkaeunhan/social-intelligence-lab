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
    
    system_prompt = "당신은 이커머스 쇼핑 플랫폼의 Senior Data Scientist입니다. 단순 키워드 일치가 아닌, 소비자의 미세한 맥락과 숨은 의도를 추론해야만 정답을 맞출 수 있는 고난도 검색 쿼리 테스트셋을 구축하고 있습니다."
    
    user_prompt = f"""
    제공된 [모든 문서에서 정독 수집된 데이터 맥락 요약본]을 완벽히 흡수하여, 실제 데이터 플랫폼 전체의 특징을 반영하면서도 변별력이 매우 높은 최종 가상 검색 쿼리(Query) 총 15개를 생성해 주세요.

    [필수 고정 교수님 요구 쿼리 4개]
    결과 JSON 배열의 1번부터 4번 항목은 아래 문장 그대로 한 글자도 바꾸지 말고 완벽히 고정하여 출력해야 합니다:
    1. "제품 평가 자체는 좋은데 별점은 낮은 리뷰를 찾아줘"
    2. "뽁뽁이 포장 상태에 만족한 리뷰"
    3. "화이트 색상에 대한 불만이 있는 리뷰"
    4. "자식들한테 사줬다는 리뷰를 찾아줘"

    [나머지 11개 쿼리 생성 조건]
    제공된 모든 문서의 마이크로 맥락 요약본을 바탕으로, 단순 키워드(단어) 일치 방식의 검색 엔진은 완전히 실패하지만, 우리가 구축할 '하이브리드 멀티 벡터 검색 + 리랭커 + 메타데이터 필터링 시스템'은 문맥을 추론하여 정답을 찾아낼 수 있는 고난도 인지적 질의(Cognitive Query) 11개를 자유롭게 생성해 주세요.

    교수님의 예시에 갇히지 말고, 실제 소비자가 이커머스 플랫폼에서 통합 검색창에 던질 법한 다음과 같은 다양한 현실적 시나리오들을 적극적으로 반영해야 합니다:

    1. 시간적 맥락 및 내구성 추론: "한 달 동안 써봤는데", "1주 차에는 좋았으나 지금은", "롱텀 사용기" 등 구매 직후가 아니라 일정 시간이 흐른 뒤 발생한 품질 변화나 내구성에 대한 소비자 VoC를 추적하는 질의
    2. 간접적 기기 성능 및 하드웨어 정황: "발열 때문에 손바닥 뜨거움", "배터리 광탈", "버벅거림", "액정 오줌액정 현상" 등 직접적인 기술 용어 대신 소비자가 체감한 직관적 불편 정황을 묘사하는 질의
    3. 가격 대 가치 및 가성비 판단: "비싸서 돈값 못함", "이 가격이면 차라리 당근에서 중고 삼", "세일할 때 사서 가성비 대박" 등 가격 대비 심리적 만족도를 추론해야 하는 질의
    4. 비교 및 경쟁사 언급 (Hard Negative 확장): 타사 제품이나 이전 세대 모델과 치열하게 비교하는 맥락 (예: "S25 쓰다가 17로 왔는데 역체감 심함", "전작에 비해 혁신이 없음")
    5. 비즈니스 서비스 정황 및 CSVoC: 쿠팡 파손 배송, 교환 신청 프로세스 불만, 상담원 대처에 대한 분노, 반품 정황 등 제품 자체의 품질이 아닌 공급망/서비스단에서의 정황 묘사 질의
    6. 바이럴/체험단 및 노이즈 문맥: "한달살기 지원 받아 작성", "무상 제공 피드" 등 대가성 노이즈의 뉘앙스가 짙게 풍기는 질의군을 골라내기 위한 검색어 정황

    LLM의 고유한 도메인 지식과 집적된 데이터 맥락 보고서를 결합하여, 질문의 형태(의문문, 구어체 단문, 서술형 장문 등)와 타겟 의도를 최대한 다양하게 분산시켜 변별력 높은 11개의 쿼리를 완성해 주세요.

    [모든 문서에서 정독 수집된 데이터 맥락 요약본]
    {combined_context}

    [출력 형식]
    반드시 아래 예시와 같이 마크다운 외벽 없이 오직 순수한 JSON 배열 형식만 반환하세요. Response는 오직 순수한 JSON 데이터여야 합니다.
    [
      {{"query_id": "Q01", "query": "제품 평가 자체는 좋은데 별점은 낮은 리뷰를 찾아줘", "intent_type": "Sentiment_Contradiction"}},
      {{"query_id": "Q02", "query": "뽁뽁이 포장 상태에 만족한 리뷰", "intent_type": "Packaging_Quality"}},
      {{"query_id": "Q03", "query": "화이트 색상에 대한 불만이 있는 리뷰", "intent_type": "Color_Dissatisfaction"}},
      {{"query_id": "Q04", "query": "자식들한테 사줬다는 리뷰를 찾아줘", "intent_type": "Purchase_Context"}},
      ... (추가 11개 생성하여 총 15개 세트 완결)
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