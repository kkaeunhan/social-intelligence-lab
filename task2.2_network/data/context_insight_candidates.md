# Step 6 Pre-LLM Context Insight Candidates

LLM 해석 전에 PMI/NPMI 기반 연관어쌍을 제품/감성/테마 단위로 선별한 후보 목록입니다.

## Summary

- Candidate count: 224
- LLM-ready candidate count: 224
- Filters: min_count=5, min_npmi=0.2

## Theme Counts

- 배송/포장 리스크: 67
- 기타 제품 인식: 54
- 카메라/화질: 31
- 가격/가성비: 23
- 불량/교환/환불: 15
- 성능/속도: 12
- 디자인/색상/마감: 9
- 배터리/충전: 8
- 무게/휴대성/그립: 4
- 데이터 이전/설정: 1

## Product-Sentiment Candidates


### galaxy_s26 / positive

- 사전 + 예약 (theme=가격/가성비, count=58, npmi=0.85482, score=2.535721, strength=strong)
  - evidence: 삼성페이로 생활이 바뀌네요 아이폰만 오래 쓰다가 이번에 갤럭시 S26으로 넘어오면서 사전예약보다는 비싸지만 쿠팡 라이브로 정가보다 조금 저렴하게 구매했어요.
- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=9, npmi=0.945764, score=1.727624, strength=medium)
  - evidence: 역시나 저도 배송은 불만족스러웠는데요 고가의 가전제품이니만큼 뽁뽁이를 좀 야무지게 싸주면 좋겠어요.
- 라이브 + 방송 (theme=기타 제품 인식, count=20, npmi=0.735774, score=1.711153, strength=strong)
  - evidence: 라이브방송때 샀는데 타사보다 조건이 좋아 만족합니다.
- ai + 기능 (theme=기타 제품 인식, count=60, npmi=0.558464, score=1.707004, strength=strong)
  - evidence: 추가된 AI 기능과 카메라 자동 성능이 좋아져서 흔들림이 거의 없이 사진이 찍힙니다.
- 버벅 + 성능 아쉽다 (theme=성능/속도, count=14, npmi=0.81, score=1.702658, strength=medium)
  - evidence: 성능 -걍 평범하게 쓰는 사람인데 다 괜찮은 것 같고, 다만 유튜브가 가끔 버벅거림.
- 무리 + 손목 (theme=무게/휴대성/그립, count=10, npmi=0.855673, score=1.619912, strength=medium)
  - evidence: 기존에 쓰던 프로 모델은 묵직해서 장시간 사용하면 손목에 무리가 많이 왔는데, S26은 손에 쥐자마자 느껴지는 가벼움 덕분에 일상적인 사용이 훨씬 편해졌습니다.
- 바이올렛 + 코발트 (theme=디자인/색상/마감, count=12, npmi=0.741405, score=1.489709, strength=medium)
  - evidence: 그런데 매장 가서 실물을 보니까 이 코발트 바이올렛이 너무 쨍하고 촌스러운 보라색이 아니라, 은은하면서도 깊이감 있는 엄청 고급스러운 색상이라서 단번에 마음을 빼앗겼어요.
- 보호 + 사생활 (theme=기타 제품 인식, count=8, npmi=0.802559, score=1.409612, strength=medium)
  - evidence: 넘나좋아요ᄏᄏ 처음으로 새핸드폰 사전구매해서 사보는거 같습니다 플립5 쓰다가 바꾼건데 생각보자 크다는 느낌이었고 깔끔하고 예쁘다는 첫인상이었습니다 :) 다만 아쉬운것은 사생활 보호 기능이 울트라에만 있다는거...ᅮᅮ 아무리 찾아봐도 없길래 검색해봤더니 울트라만 되는거...

### galaxy_s26_ultra / negative

- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=5, npmi=1.0, score=1.277863, strength=exploratory)
  - evidence: 이백만원짜리 휴대폰인데 뾱뾱이도 없이 배송됨
- 서비스 + 센터 (theme=불량/교환/환불, count=6, npmi=0.871965, score=1.215959, strength=exploratory)
  - evidence: 고객센터에 문의하니 모두가 하나 같이 고객 잘못으로 넘기시는데 기만 아닙니까?

### galaxy_s26_ultra / positive

- 사전 + 예약 (theme=가격/가성비, count=62, npmi=0.822009, score=2.478598, strength=strong)
  - evidence: 그래도 사전예약으로 용량 업그레이드까지 받아서 전체적으로는 만족도가 높은 구매였음.
- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=11, npmi=0.931612, score=1.819676, strength=medium)
  - evidence: 뽁뽁이 달랑 한겹붙여 박스에 넣어 왔는데...백오십만원도 넘는 폰을...작동 잘 되길 바래야 하는지...
- 보호 + 사생활 (theme=기타 제품 인식, count=22, npmi=0.732069, score=1.747964, strength=strong)
  - evidence: 제가 옆에서 본 느낌은 사생활보호 기능인데 좀 신기했어요~ 사생활보호 필름 사용하면 눈이 좀 불편한데 확실히 그것보다는 덜 피곤하다고 느꼈어요.
- 공간 + 저장 (theme=기타 제품 인식, count=13, npmi=0.806256, score=1.66036, strength=medium)
  - evidence: 저장공간 및 활용성 (512GB) 512GB 용량이라 매우 여유로운 제품입니다.
- 버벅 + 성능 아쉽다 (theme=성능/속도, count=12, npmi=0.76422, score=1.537029, strength=medium)
  - evidence: 훨씬 카메라 기능이 향상 되었네요 프로세싱도 진짜 빨라서 버벅거리는 딜레이가 없어요 게임 전화 s
- 불량 + 불량 의심 (theme=불량/교환/환불, count=6, npmi=0.930385, score=1.470343, strength=exploratory)
  - evidence: (호ᅡ이트나 핑크 살걸 싶긴함 ᄒᄒ) 그외 아직 큰 불량 없이 잘 쓰고 있어요 평소엔 프라이버시 기능은 쓰지 않고 지하철 탈 때만 이용할 생각이에요 배터리도 빠르게 충전되고 천천히 떨어져서 만족해요 가격은 잇지만 알뜰요금제로 야무지게 써볼게요~!
- 더블 + 스토리 (theme=기타 제품 인식, count=6, npmi=0.893083, score=1.411686, strength=exploratory)
  - evidence: 그동안 사전예약 없이 구매하다가 이번에 사전예약으로 더블업 스토리지로 구매했습니다.
- 바이올렛 + 코발트 (theme=디자인/색상/마감, count=8, npmi=0.789085, score=1.388152, strength=medium)
  - evidence: 먼저 디자인부터 말씀드리면 코발트 바이올렛 색상이 정말 고급스럽습니다.

### galaxy_z_flip7 / negative

- 배송 + 포장 (theme=배송/포장 리스크, count=5, npmi=0.450108, score=0.6144, strength=exploratory)
  - evidence: 배송 진짜 욕나오네 포장큰건 분실방지때문에 쿠팡시스템을 이해를 했는데 이거는 도통 이해가 안된다 어떻게 전자제품사는데 몇만원몇십만원도 아니고 박스안에 뽁뽁이를 하나도 안넣어두고 배송하고 사진보면 개봉전에 상자안에 구멍난 흔적도있다 고의적으로 어떤의도로 구멍이 뚫린지 모르겠으나 소비자입장에선 매우 언짢네

### galaxy_z_flip7 / positive

- 사전 + 예약 (theme=가격/가성비, count=50, npmi=0.880238, score=2.528506, strength=strong)
  - evidence: ...성능, 배터리는 써봐야 알거같고 디자인은 최고입니다 무게는 생각보다 무겁드라구요 (아이폰 15프로보다 조금 더 무거움) 이번 쿠팡 사전예약 혜택 이것저것 신경 많이 쓴것 같았어요 전체적으로 만족스럽고 다른 하자는 없었으나 배송상태에서 실망스러웠네요 이런 포장, 배송 후기를 봤는데 저도 겪을 줄이야 ᄏᄏᄏ 신경써주세요~!
- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=13, npmi=1.0, score=2.052036, strength=medium)
  - evidence: 너무 이쁘구요 ✅️배송은 구매하고 나서 생각보다 오래걸렸는데,, 비싼 고가 제품인데 뽁뽁이 없이 에어비닐?
- 박스 + 포장 (theme=배송/포장 리스크, count=10, npmi=0.796412, score=1.509045, strength=medium)
  - evidence: 다만...포장...아쉽네요...에어백 2개인데..핸드폰 위쪽으로 있어 핸드폰이 박스 바닥에 있고..에어백1개는 바람이 빵빵하지않고..
- 고속 + 고속 충전 (theme=배터리/충전, count=6, npmi=0.924954, score=1.459882, strength=exploratory)
  - evidence: 배터리는 하루 종일 무난히 사용 가능하고, 고속 충전 지원되니.뭐.
- 블루 + 쉐도우 (theme=기타 제품 인식, count=9, npmi=0.758602, score=1.388662, strength=medium)
  - evidence: 사전예약후 구매한 Z플립7 블루쉐도우!
- 보호 + 필름 (theme=기타 제품 인식, count=10, npmi=0.723891, score=1.373237, strength=medium)
  - evidence: [설정]-[배터리검색]-[배터리보호]기능에서 충전 방식을 선택할 수 있어요.
- 레드 + 코랄 (theme=기타 제품 인식, count=15, npmi=0.598686, score=1.291739, strength=medium)
  - evidence: ▶️삼성전자 갤럭시 Z플립7, 코랄 레드, 512GB◀️ 리뷰해보려고 합니다.
- 사진 + 찍다 (theme=카메라/화질, count=21, npmi=0.531167, score=1.263508, strength=strong)
  - evidence: 그래서 사진용으로 아이폰 들고다님...

### galaxy_z_fold7 / negative

- 배송 + 포장 (theme=배송/포장 리스크, count=6, npmi=0.686036, score=0.978379, strength=exploratory)
  - evidence: 배송이 최악입니다.

### galaxy_z_fold7 / positive

- 사전 + 예약 (theme=가격/가성비, count=39, npmi=0.921217, score=2.498454, strength=strong)
  - evidence: 갤럭시 폴드7을 사전예약으로 구매해서 사용하고 있는데, 진짜 만족도가 높아요.
- 블루 + 쉐도우 (theme=기타 제품 인식, count=14, npmi=0.707756, score=1.486972, strength=medium)
  - evidence: 블루 쉐도우 색감 최고, 3성최고 폴드 7 대만족!
- 두께 + 무게 (theme=무게/휴대성/그립, count=40, npmi=0.43589, score=1.231491, strength=medium)
  - evidence: 폴드5와 체감되는 차이점은 무게와 두께 정도?
- 공간 + 저장 (theme=기타 제품 인식, count=7, npmi=0.725212, score=1.211786, strength=exploratory)
  - evidence: - 512GB의 넉넉한 저장 공간 사진과 영상을 자주 찍는 저에게 512GB는 부족함이 없는 용량입니다.
- ai + 구독 (theme=기타 제품 인식, count=8, npmi=0.676843, score=1.187782, strength=medium)
  - evidence: 그래서 지금 ,AI 구독 클럽 또한 해지하려고 합니다.
- 도착 + 배송 (theme=배송/포장 리스크, count=18, npmi=0.500589, score=1.141403, strength=medium)
  - evidence: 배송: 우선 배송은 전날 오후 2시 40분쯤 주문 - 다음 날 1시 10분쯤 도착했고 택배 포장은 사진 첨부했습니다.
- 무게 가볍다 + 카메라 만족 (theme=카메라/화질, count=133, npmi=0.269324, score=1.12059, strength=exploratory)
  - evidence: 무게: 가벼워요
- 넣다 + 주머니 (theme=기타 제품 인식, count=5, npmi=0.750174, score=1.099039, strength=exploratory)
  - evidence: ...고릴라 글래스 적용이 되있다고 하는데 불안하니깐 전 필름 붙였습니다 ᄏᄏ 무게는 215g인데 너무 가벼워서 주머니에 넣고 다니기 부담없더라구요 예전 폴드3는 주머니에 넣으면 부담스러웠거든요 배터리는 확실히 오래 쓸수 있더라구요 뭐 처음이라 그런거 일수도 있겠지만 하루까진 아니더라도 아침 9시부터 오후 5시정도는 쓸수 있는...

### galaxy_z_fold7 / neutral

- 무게 가볍다 + 배송 (theme=배송/포장 리스크, count=5, npmi=0.284107, score=0.461121, strength=exploratory)
  - evidence: 기기는 최고 배송은 최악 기기는 폴드4를 쓰다 폴드7로 넘어 온거라 변화가 눈에 확 들어와 생각 이상으로 좋았습니다.

### iphone_17 / negative

- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=16, npmi=1.0, score=2.011917, strength=medium)
  - evidence: 너무해요 전자기기를 이렇게 포장하시면 어떡합니까 12프로도 쿠팡에서 구매했었고 이렇게까지 엉망으로 오진 않았는데 찌그러진 박스에 뽁뽁이 하나 안감싸진 아이폰 상자보고 너무 어이없었네요...
- 가성비 나쁘다 + 배터리 짧음 (theme=가격/가성비, count=7, npmi=0.772405, score=1.179117, strength=exploratory)
  - evidence: 디자인: 별로예요
- 디자인 불만 + 배터리 짧음 (theme=배터리/충전, count=5, npmi=0.855118, score=1.135687, strength=exploratory)
  - evidence: 디자인: 별로예요
- 디자인 불만 + 무게 무겁다 (theme=디자인/색상/마감, count=5, npmi=0.746904, score=0.995989, strength=exploratory)
  - evidence: 디자인: 별로예요
- 무게 무겁다 + 카메라 불만 (theme=카메라/화질, count=5, npmi=0.746904, score=0.995989, strength=exploratory)
  - evidence: 대단하네?
- 가성비 나쁘다 + 무게 무겁다 (theme=가격/가성비, count=7, npmi=0.645856, score=0.993997, strength=exploratory)
  - evidence: 디자인: 별로예요
- 가성비 나쁘다 + 디자인 불만 (theme=가격/가성비, count=5, npmi=0.660497, score=0.884647, strength=exploratory)
  - evidence: 디자인: 별로예요
- 무게 무겁다 + 배터리 짧음 (theme=배터리/충전, count=5, npmi=0.602022, score=0.809415, strength=exploratory)
  - evidence: 디자인: 별로예요

### iphone_17 / positive

- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=22, npmi=0.984112, score=2.332958, strength=strong)
  - evidence: 포장에 신경 좀 쓰세요 아이폰17 제품 깨끗하고 흠 없어요 케이스도 보관하는데 뽁뽁이라도 하나 말아서 배송되면 좋겠어요 백만원 넘는 제품인데 그냥 박스 포장만 와서 놀랐어요 가성비: 적당한 편이에요
- 미스트 + 블루 (theme=기타 제품 인식, count=11, npmi=0.930919, score=1.814894, strength=medium)
  - evidence: 미스트블루 짱예뻐요 세이지랑 고민했는데 미스트블루 색상 너무 예뻐요!!
- 사전 + 예약 (theme=가격/가성비, count=11, npmi=0.863302, score=1.684249, strength=medium)
  - evidence: 갤럭시 사전예약 유혹을 뿌리치고 샀어요ᅮᅮ 고장안나서 오래오래 쓰고싶어요 ᄒᄒ 가성비: 적당한 편이에요
- 버벅 + 성능 아쉽다 (theme=성능/속도, count=12, npmi=0.716484, score=1.439782, strength=medium)
  - evidence: 기존 폰에서는 버벅이던 앱들이 이제는 부드럽게 실행되고, 화면 전환도 빠릿빠릿해서 사용할 때마다 만족감이 커요.
- 불량 + 불량 의심 (theme=불량/교환/환불, count=6, npmi=0.901037, score=1.421482, strength=exploratory)
  - evidence: 크기 전부 만족하나 불량 상품..
- 이자 + 할부 (theme=기타 제품 인식, count=5, npmi=0.957396, score=1.404092, strength=exploratory)
  - evidence: 무이자 할부까지 활용하면 부담도 줄고 만족도는 훨씬 높습니다 가성비: 성능에 비해 저렴해요
- 고장 + 나다 (theme=불량/교환/환불, count=10, npmi=0.737889, score=1.398638, strength=medium)
  - evidence: 갤럭시 사전예약 유혹을 뿌리치고 샀어요ᅮᅮ 고장안나서 오래오래 쓰고싶어요 ᄒᄒ 가성비: 적당한 편이에요
- 자유 + 통신사 (theme=기타 제품 인식, count=10, npmi=0.693664, score=1.315949, strength=medium)
  - evidence: 특히 자급제로 구매하면 통신사에 구애받지 않고 저렴한 가격으로 자유롭게 사용할 수 있어 만족스럽습니다 가격대가 높은 편이라 부담이 있을 수 있지만, 장기간 사용할 것을 고려하면 충분히 가치 있는 선택이라고 생각됩니다.

### iphone_17 / neutral

- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=8, npmi=1.0, score=1.558759, strength=medium)
  - evidence: 택배박스가 구겨져왔지만, 내부 뽁뽁이포장으로 본품에 손상 없이 받았습니다.
- 배송 + 뽁뽁이 (theme=배송/포장 리스크, count=6, npmi=0.575754, score=0.833974, strength=exploratory)
  - evidence: 배송일보다 빨리왔어요 10/22 배송일인데 10/11 에 받았어요.
- 배송 + 완충 포장 (theme=배송/포장 리스크, count=6, npmi=0.575754, score=0.833974, strength=exploratory)
  - evidence: 배송일보다 빨리왔어요 10/22 배송일인데 10/11 에 받았어요.
- 박스 + 배송 (theme=배송/포장 리스크, count=6, npmi=0.462659, score=0.68868, strength=exploratory)
  - evidence: 근데 왜 본품박스에 오염물이 묻은건지, 궁금하네요.
- 박스 + 뽁뽁이 (theme=배송/포장 리스크, count=6, npmi=0.334443, score=0.524114, strength=exploratory)
  - evidence: 택배박스가 구겨져왔지만, 내부 뽁뽁이포장으로 본품에 손상 없이 받았습니다.
- 박스 + 완충 포장 (theme=배송/포장 리스크, count=6, npmi=0.334443, score=0.524114, strength=exploratory)
  - evidence: 근데 왜 본품박스에 오염물이 묻은건지, 궁금하네요.

### iphone_17_pro / negative

- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=20, npmi=1.0, score=2.161625, strength=strong)
  - evidence: 아니 한두푼하는것도 아니고 살다살다 백만원짜리를 배송ᄋ 이따구로 주는데는 첨봄 충전재 뽁뽁이 하나 없이 상자안에 저거 하나 달랑 보내는게 맞나요?
- 가성비 나쁘다 + 무게 무겁다 (theme=가격/가성비, count=8, npmi=0.549975, score=0.904005, strength=medium)
  - evidence: 디자인: 별로예요
- 뽁뽁이 + 포장 (theme=배송/포장 리스크, count=12, npmi=0.326413, score=0.65529, strength=medium)
  - evidence: 뭐 이딴 포장에 이딴 배송이 다 있을까싶네 뭐 포장을 이딴식으로 해서 보내지??
- 완충 포장 + 포장 (theme=배송/포장 리스크, count=12, npmi=0.326413, score=0.65529, strength=medium)
  - evidence: 뭐 이딴 포장에 이딴 배송이 다 있을까싶네 뭐 포장을 이딴식으로 해서 보내지??
- 넣다 + 뽁뽁이 (theme=배송/포장 리스크, count=6, npmi=0.422413, score=0.629268, strength=exploratory)
  - evidence: 성의가 없어도 너무 없고 제품 하자라도 생기면 지들이 어쩌려고 이딴식으로 배송을 할까 싶네 뭐 예의상 뽁뽁이로라도 한번 싸서 보내던가 비닐이라도 하나 넣어보내는게 맞지않나 어떻게된게 아이폰상자 하나 딸랑 오냐 ᄏᄏᄏᄏ역시 쿠팡이 쿠팡하는구나 이거 주말에 받았으니 평일에 핸드폰 검사해보고 문제 생기면 가만안있지 또 가성비:...
- 넣다 + 완충 포장 (theme=배송/포장 리스크, count=6, npmi=0.422413, score=0.629268, strength=exploratory)
  - evidence: 디자인: 보통이에요
- 문제 + 포장 (theme=배송/포장 리스크, count=8, npmi=0.355498, score=0.603521, strength=medium)
  - evidence: 200만 짜리 상품 포장 박스는 다 터져서 왔어요!!
- 맞다 + 배송 (theme=배송/포장 리스크, count=6, npmi=0.381879, score=0.572878, strength=exploratory)
  - evidence: 아니 한두푼하는것도 아니고 살다살다 백만원짜리를 배송ᄋ 이따구로 주는데는 첨봄 충전재 뽁뽁이 하나 없이 상자안에 저거 하나 달랑 보내는게 맞나요?

### iphone_17_pro / positive

- 버벅 + 성능 아쉽다 (theme=성능/속도, count=18, npmi=0.832215, score=1.88071, strength=medium)
  - evidence: 동영상 촬영하고 여행시 길게 써도 버벅이는 느낌 없어요.
- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=8, npmi=0.96945, score=1.703862, strength=medium)
  - evidence: 에아스토어 모두 품절인데 쿠팡 로켓배송으로 바로 다음날 받아볼수있어서 바로 주문했어요 근데 뽁뽁이도 없이 누가 반품한 제품 받은것마냥 찢어진 완충재만 덩그러니..
- 사전 + 예약 (theme=가격/가성비, count=9, npmi=0.824648, score=1.511063, strength=medium)
  - evidence: 급하게 쿠팡구매를 하게됐네요~ 고민말고 나왔을떄 사전구매예약할걸그랬네여 ᄒ 전반적으로 밧데리 사용량도 너무 좋고, 기존에 쓰던건 반날절만되도...또충전했어야했는데 역시 새폰이다보니 ᄏᄏ역시나 오래가네요~ 카메라같은 경우도 xr도 내 나름 만족했는데..
- 오렌지 + 코스믹 (theme=기타 제품 인식, count=12, npmi=0.734408, score=1.478707, strength=medium)
  - evidence: 가장 인상 깊었던 건 역시 코스믹 오렌지 색상입니다.
- 공간 + 저장 (theme=기타 제품 인식, count=11, npmi=0.656354, score=1.288271, strength=medium)
  - evidence: 아이폰 14 Pro에서 넘어왔는데, 전반적인 사용 경험이 더 쾌적해졌고 특히 저장 공간 스트레스가 사라져서 만족이에요 ᄒᄒ 가성비: 적당한 편이에요
- 영상 + 촬영 (theme=카메라/화질, count=50, npmi=0.418432, score=1.253086, strength=medium)
  - evidence: 다만 게임이나 영상 촬영을 많이 하면 소모는 빠른 편이다.
- 성능 + 카메라 (theme=카메라/화질, count=157, npmi=0.294264, score=1.252018, strength=exploratory)
  - evidence: 카메라 성능: 보통이에요
- 사진 + 찍다 (theme=카메라/화질, count=84, npmi=0.333486, score=1.161799, strength=medium)
  - evidence: 사진기능 만족!밧데리 굿.

### iphone_17_pro / neutral

- 박스 + 배송 (theme=배송/포장 리스크, count=6, npmi=0.5, score=0.729662, strength=exploratory)
  - evidence: 약 1km 거리에서도 깨짐 없이 선명해서 요즘은 그냥 망원경 대신 아이폰 들고 다닙니다^^ 결론은 폰은 살아있었고, 카메라는 예술이고, 배송 박스는...
- 배송 + 카메라 만족 (theme=배송/포장 리스크, count=5, npmi=0.250566, score=0.366537, strength=exploratory)
  - evidence: 17일 사전예약, 25일 배송완료 휴우~ "10월 1일 도착보장" 글자가 언제 바뀌나 수시로 확인하니 속이 타들어 감ᅮ (결국 예정일 보다는 일찍 받음) 5s> L*> 삼*> 17pro 돌고 돌아 10여년만에 다시 옴.

### iphone_17_pro_max / negative

- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=8, npmi=0.888469, score=1.396049, strength=medium)
  - evidence: 200만원이 넘는 고가의 물건을 뽁뽁이 하나 없이 배송하는 깡이 대단해요!!
- 넘다 + 뽁뽁이 (theme=배송/포장 리스크, count=5, npmi=0.572542, score=0.759447, strength=exploratory)
  - evidence: 200만원이 넘는 고가의 물건을 뽁뽁이 하나 없이 배송하는 깡이 대단해요!!
- 넘다 + 완충 포장 (theme=배송/포장 리스크, count=5, npmi=0.495361, score=0.665302, strength=exploratory)
  - evidence: 디자인: 보통이에요
- 완충 포장 + 포장 (theme=배송/포장 리스크, count=8, npmi=0.358556, score=0.626047, strength=medium)
  - evidence: 애플의 포장 기술력에 감탄중입니다!
- 뽁뽁이 + 포장 (theme=배송/포장 리스크, count=7, npmi=0.305069, score=0.51325, strength=exploratory)
  - evidence: 애플의 포장 기술력에 감탄중입니다!
- 박스 + 완충 포장 (theme=배송/포장 리스크, count=5, npmi=0.306847, score=0.435816, strength=exploratory)
  - evidence: 진짜 ᄉ 벌 쿠팡 관리좀 쳐해라 200만원짜리를 팔면 상품 관리좀 해라 선물주는건데 아이폰 박스가 찌그러져있질 않나 떼가 껴있지 않나 포장은 ᄉ벌 뽁뽁이에 테이프한번 안감고 열어보니까 혼자 나뒹굴고 있고 이럴거면 뭐하러 포장하냐 걍던져주지
- 배송 + 완충 포장 (theme=배송/포장 리스크, count=6, npmi=0.247172, score=0.399724, strength=exploratory)
  - evidence: 200만원이 넘는 고가의 물건을 뽁뽁이 하나 없이 배송하는 깡이 대단해요!!

### iphone_17_pro_max / positive

- 사전 + 예약 (theme=가격/가성비, count=11, npmi=0.76944, score=1.483376, strength=medium)
  - evidence: 뭔가 사전예약 성공한 사람만의 특권 같은 느낌이랄까요.
- 배송 + 빠르다 (theme=배송/포장 리스크, count=31, npmi=0.526488, score=1.373054, strength=strong)
  - evidence: 배송 빠르고 너무 좋아요 1테라 구하기 어렵다고 해서 쿠팡에서 구매 했는데, 배송고 빠르고 너무 좋아요 사용 시간: 아주길어요
- 성능 + 카메라 (theme=카메라/화질, count=46, npmi=0.472505, score=1.369497, strength=strong)
  - evidence: 카메라 성능: 아주뛰어나요
- 용량 + 저장 (theme=기타 제품 인식, count=9, npmi=0.736788, score=1.330602, strength=medium)
  - evidence: -저장 공간: 2TB 덕분에 사진, 영상, 대용량 앱을 아무 걱정 없이 담을 수 있어 든든합니다.
- 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=5, npmi=0.916337, score=1.325588, strength=exploratory)
  - evidence: ...었는데 아이폰 17 프로맥스와 함께 할 수 있어 더 좋은 것 같습니다 다만, 남편과 함께 주문했는데 제꺼는 뽁뽁이가 싸여오지 않고 남편꺼는 뽁뽁이에 싸여왔더라고요ᅲᅲ 조금 더 안전하게 배송되었다면 좋았겠다 싶은 아쉬움이 살짝 있지만 그래도 핸드폰 자체에 문제가 있거나 아쉬움이 있는건 아니라서 만족하고 잘 사용할 것 같습니...
- 버벅 + 성능 아쉽다 (theme=성능/속도, count=5, npmi=0.883135, score=1.277831, strength=exploratory)
  - evidence: 14에서는 역광이나 조명이 복잡한 환경에서 살짝 버벅이는 느낌이 있었는데, 이제는 프레임을 바꿔도 거의 즉각적으로 따라와요.
- 사진 + 영상 (theme=카메라/화질, count=31, npmi=0.481895, score=1.263311, strength=strong)
  - evidence: 디테일도 잘 살아 있어서 일상 사진뿐만 아니라 풍경이나 야간 촬영에서도 만족도가 높은 편이었어요.
- 상태 + 포장 (theme=배송/포장 리스크, count=8, npmi=0.723298, score=1.254354, strength=medium)
  - evidence: 선물은 포장 상태가 절반이라고 생각하는 편이라 이 부분은 매우 만족스러웠습니다.

## Global/Comparative Candidates

- [global] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=145, npmi=0.972495)
- [global] 사전 + 예약 (theme=가격/가성비, count=250, npmi=0.865979)
- [global] 불량 + 불량 의심 (theme=불량/교환/환불, count=41, npmi=0.890414)
- [global] 버벅 + 성능 아쉽다 (theme=성능/속도, count=64, npmi=0.753139)
- [global] 라이브 + 방송 (theme=기타 제품 인식, count=29, npmi=0.795821)
- [global] 공간 + 저장 (theme=기타 제품 인식, count=54, npmi=0.682552)
- [global] 고속 + 고속 충전 (theme=배터리/충전, count=19, npmi=0.8514)
- [global] 바이올렛 + 코발트 (theme=디자인/색상/마감, count=20, npmi=0.826441)
- [global] 이자 + 할부 (theme=기타 제품 인식, count=17, npmi=0.818329)
- [global] 보호 + 사생활 (theme=기타 제품 인식, count=30, npmi=0.698728)
- [global] 사진 + 찍다 (theme=카메라/화질, count=271, npmi=0.462428)
- [global] 박스 + 포장 (theme=배송/포장 리스크, count=92, npmi=0.541545)
- [product/galaxy_s26] 사전 + 예약 (theme=가격/가성비, count=60, npmi=0.852466)
- [product/galaxy_s26] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=12, npmi=0.955353)
- [product/galaxy_s26] ai + 기능 (theme=기타 제품 인식, count=60, npmi=0.568138)
- [product/galaxy_s26] 버벅 + 성능 아쉽다 (theme=성능/속도, count=14, npmi=0.812326)
- [product/galaxy_s26] 라이브 + 방송 (theme=기타 제품 인식, count=20, npmi=0.73121)
- [product/galaxy_s26] 무리 + 손목 (theme=무게/휴대성/그립, count=10, npmi=0.857276)
- [product/galaxy_s26] 바이올렛 + 코발트 (theme=디자인/색상/마감, count=12, npmi=0.74443)
- [product/galaxy_s26] 보호 + 사생활 (theme=기타 제품 인식, count=8, npmi=0.804626)
- [product/galaxy_s26] 불량 + 불량 의심 (theme=불량/교환/환불, count=5, npmi=0.957875)
- [product/galaxy_s26] 녹음 + 통화 (theme=데이터 이전/설정, count=6, npmi=0.839602)
- [product/galaxy_s26] 사진 + 찍다 (theme=카메라/화질, count=36, npmi=0.469132)
- [product/galaxy_s26] gb + 영상 (theme=카메라/화질, count=21, npmi=0.497514)
- [product/galaxy_s26_ultra] 사전 + 예약 (theme=가격/가성비, count=64, npmi=0.810822)
- [product/galaxy_s26_ultra] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=17, npmi=0.948582)
- [product/galaxy_s26_ultra] 보호 + 사생활 (theme=기타 제품 인식, count=22, npmi=0.738503)
- [product/galaxy_s26_ultra] 불량 + 불량 의심 (theme=불량/교환/환불, count=9, npmi=0.903147)
- [product/galaxy_s26_ultra] 공간 + 저장 (theme=기타 제품 인식, count=13, npmi=0.794427)
- [product/galaxy_s26_ultra] 서비스 + 센터 (theme=불량/교환/환불, count=7, npmi=0.937917)
- [product/galaxy_s26_ultra] 버벅 + 성능 아쉽다 (theme=성능/속도, count=12, npmi=0.768904)
- [product/galaxy_s26_ultra] 더블 + 스토리 (theme=기타 제품 인식, count=6, npmi=0.894857)
- [product/galaxy_s26_ultra] 바이올렛 + 코발트 (theme=디자인/색상/마감, count=8, npmi=0.79284)
- [product/galaxy_s26_ultra] 고속 + 고속 충전 (theme=배터리/충전, count=7, npmi=0.82877)
- [product/galaxy_s26_ultra] 인식 + 지문 (theme=기타 제품 인식, count=6, npmi=0.835051)
- [product/galaxy_s26_ultra] 사진 + 찍다 (theme=카메라/화질, count=50, npmi=0.420171)
- [product/galaxy_z_flip7] 사전 + 예약 (theme=가격/가성비, count=50, npmi=0.884815)
- [product/galaxy_z_flip7] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=19, npmi=1.0)
- [product/galaxy_z_flip7] 박스 + 포장 (theme=배송/포장 리스크, count=14, npmi=0.701024)
- [product/galaxy_z_flip7] 고속 + 고속 충전 (theme=배터리/충전, count=6, npmi=0.926369)
- [product/galaxy_z_flip7] 블루 + 쉐도우 (theme=기타 제품 인식, count=9, npmi=0.763644)
- [product/galaxy_z_flip7] 보호 + 필름 (theme=기타 제품 인식, count=11, npmi=0.711489)
- [product/galaxy_z_flip7] 레드 + 코랄 (theme=기타 제품 인식, count=15, npmi=0.608377)
- [product/galaxy_z_flip7] 사진 + 찍다 (theme=카메라/화질, count=21, npmi=0.526601)
- [product/galaxy_z_flip7] 공간 + 저장 (theme=기타 제품 인식, count=7, npmi=0.734433)
- [product/galaxy_z_flip7] 박스 + 뽁뽁이 (theme=배송/포장 리스크, count=9, npmi=0.604822)
- [product/galaxy_z_flip7] 박스 + 완충 포장 (theme=배송/포장 리스크, count=9, npmi=0.604822)
- [product/galaxy_z_flip7] 영상 + 저장 (theme=카메라/화질, count=8, npmi=0.626129)
- [product/galaxy_z_fold7] 사전 + 예약 (theme=가격/가성비, count=40, npmi=0.924464)
- [product/galaxy_z_fold7] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=6, npmi=0.962551)
- [product/galaxy_z_fold7] 블루 + 쉐도우 (theme=기타 제품 인식, count=14, npmi=0.713525)
- [product/galaxy_z_fold7] 불량 + 불량 의심 (theme=불량/교환/환불, count=5, npmi=0.957586)
- [product/galaxy_z_fold7] 두께 + 무게 (theme=무게/휴대성/그립, count=41, npmi=0.456222)
- [product/galaxy_z_fold7] 공간 + 저장 (theme=기타 제품 인식, count=7, npmi=0.729688)
- [product/galaxy_z_fold7] ai + 구독 (theme=기타 제품 인식, count=8, npmi=0.682291)
- [product/galaxy_z_fold7] 무게 가볍다 + 카메라 만족 (theme=카메라/화질, count=139, npmi=0.283979)
- [product/galaxy_z_fold7] 박스 + 포장 (theme=배송/포장 리스크, count=8, npmi=0.638937)
- [product/galaxy_z_fold7] tb + 저장 (theme=기타 제품 인식, count=6, npmi=0.688067)
- [product/galaxy_z_fold7] 부모 + 선물 (theme=기타 제품 인식, count=9, npmi=0.585596)
- [product/galaxy_z_fold7] 블루 + 색상 (theme=디자인/색상/마감, count=13, npmi=0.516819)
- [product/iphone_17] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=46, npmi=0.990368)
- [product/iphone_17] 미스트 + 블루 (theme=기타 제품 인식, count=11, npmi=0.934173)
- [product/iphone_17] 불량 + 불량 의심 (theme=불량/교환/환불, count=10, npmi=0.910486)
- [product/iphone_17] 사전 + 예약 (theme=가격/가성비, count=12, npmi=0.857174)
- [product/iphone_17] 버벅 + 성능 아쉽다 (theme=성능/속도, count=12, npmi=0.730164)
- [product/iphone_17] 이자 + 할부 (theme=기타 제품 인식, count=5, npmi=0.959047)
- [product/iphone_17] 고장 + 나다 (theme=불량/교환/환불, count=11, npmi=0.717992)
- [product/iphone_17] 자유 + 통신사 (theme=기타 제품 인식, count=10, npmi=0.707728)
- [product/iphone_17] 약정 + 통신사 (theme=기타 제품 인식, count=11, npmi=0.660313)
- [product/iphone_17] 박스 + 포장 (theme=배송/포장 리스크, count=23, npmi=0.519809)
- [product/iphone_17] 사진 + 찍다 (theme=카메라/화질, count=52, npmi=0.414617)
- [product/iphone_17] 종일 + 하루 (theme=기타 제품 인식, count=8, npmi=0.684276)
- [product/iphone_17_pro] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=30, npmi=0.976073)
- [product/iphone_17_pro] 버벅 + 성능 아쉽다 (theme=성능/속도, count=20, npmi=0.831349)
- [product/iphone_17_pro] 사전 + 예약 (theme=가격/가성비, count=11, npmi=0.852752)
- [product/iphone_17_pro] 성능 + 카메라 (theme=카메라/화질, count=158, npmi=0.404498)
- [product/iphone_17_pro] 오렌지 + 코스믹 (theme=기타 제품 인식, count=12, npmi=0.740485)
- [product/iphone_17_pro] 영상 + 촬영 (theme=카메라/화질, count=50, npmi=0.472453)
- [product/iphone_17_pro] 사진 + 찍다 (theme=카메라/화질, count=85, npmi=0.396746)
- [product/iphone_17_pro] 사진 + 영상 (theme=카메라/화질, count=98, npmi=0.369044)
- [product/iphone_17_pro] 공간 + 저장 (theme=기타 제품 인식, count=11, npmi=0.6601)
- [product/iphone_17_pro] 사진 + 카메라 (theme=카메라/화질, count=154, npmi=0.305503)
- [product/iphone_17_pro] 박스 + 포장 (theme=배송/포장 리스크, count=23, npmi=0.508008)
- [product/iphone_17_pro] 불량 + 불량 의심 (theme=불량/교환/환불, count=6, npmi=0.774536)
- [product/iphone_17_pro_max] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=15, npmi=0.937428)
- [product/iphone_17_pro_max] 사전 + 예약 (theme=가격/가성비, count=13, npmi=0.779817)
- [product/iphone_17_pro_max] 성능 + 카메라 (theme=카메라/화질, count=46, npmi=0.495451)
- [product/iphone_17_pro_max] 용량 + 저장 (theme=기타 제품 인식, count=9, npmi=0.744855)
- [product/iphone_17_pro_max] 배송 + 빠르다 (theme=배송/포장 리스크, count=33, npmi=0.486747)
- [product/iphone_17_pro_max] 버벅 + 성능 아쉽다 (theme=성능/속도, count=5, npmi=0.886207)
- [product/iphone_17_pro_max] 사진 + 영상 (theme=카메라/화질, count=31, npmi=0.48417)
- [product/iphone_17_pro_max] 영상 + 촬영 (theme=카메라/화질, count=16, npmi=0.560567)
- [product/iphone_17_pro_max] 영상 + 화면 (theme=카메라/화질, count=20, npmi=0.514214)
- [product/iphone_17_pro_max] 선물 + 주다 (theme=기타 제품 인식, count=7, npmi=0.688089)
- [product/iphone_17_pro_max] 사진 + 카메라 (theme=카메라/화질, count=41, npmi=0.390911)
- [product/iphone_17_pro_max] 상태 + 포장 (theme=배송/포장 리스크, count=11, npmi=0.573061)
- [sentiment/negative] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=55, npmi=0.985877)
- [sentiment/negative] 불량 + 불량 의심 (theme=불량/교환/환불, count=11, npmi=0.849126)
- [sentiment/negative] 디자인 불만 + 무게 무겁다 (theme=디자인/색상/마감, count=14, npmi=0.707288)
- [sentiment/negative] 서비스 + 센터 (theme=불량/교환/환불, count=10, npmi=0.775943)
- [sentiment/negative] 가성비 나쁘다 + 무게 무겁다 (theme=가격/가성비, count=21, npmi=0.612495)
- [sentiment/negative] 무게 무겁다 + 카메라 불만 (theme=카메라/화질, count=13, npmi=0.683044)
- [sentiment/negative] 디자인 불만 + 배터리 짧음 (theme=배터리/충전, count=11, npmi=0.719263)
- [sentiment/negative] 디자인 불만 + 카메라 불만 (theme=카메라/화질, count=11, npmi=0.719263)
- [sentiment/negative] 가성비 나쁘다 + 배터리 짧음 (theme=가격/가성비, count=16, npmi=0.6156)
- [sentiment/negative] 사전 + 예약 (theme=가격/가성비, count=6, npmi=0.826394)
- [sentiment/negative] 스티커 + 취급 (theme=기타 제품 인식, count=6, npmi=0.801473)
- [sentiment/negative] 배터리 짧음 + 카메라 불만 (theme=카메라/화질, count=9, npmi=0.627111)
- [sentiment/positive] 사전 + 예약 (theme=가격/가성비, count=240, npmi=0.865672)
- [sentiment/positive] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=71, npmi=0.961684)
- [sentiment/positive] 버벅 + 성능 아쉽다 (theme=성능/속도, count=62, npmi=0.746132)
- [sentiment/positive] 불량 + 불량 의심 (theme=불량/교환/환불, count=26, npmi=0.884432)
- [sentiment/positive] 라이브 + 방송 (theme=기타 제품 인식, count=28, npmi=0.794023)
- [sentiment/positive] 공간 + 저장 (theme=기타 제품 인식, count=54, npmi=0.682045)
- [sentiment/positive] 고속 + 고속 충전 (theme=배터리/충전, count=19, npmi=0.848094)
- [sentiment/positive] 바이올렛 + 코발트 (theme=디자인/색상/마감, count=20, npmi=0.822537)
- [sentiment/positive] 보호 + 사생활 (theme=기타 제품 인식, count=30, npmi=0.707393)
- [sentiment/positive] 이자 + 할부 (theme=기타 제품 인식, count=16, npmi=0.81565)
- [sentiment/positive] 사진 + 찍다 (theme=카메라/화질, count=270, npmi=0.447856)
- [sentiment/positive] 레드 + 코랄 (theme=기타 제품 인식, count=15, npmi=0.74895)
- [sentiment/neutral] 뽁뽁이 + 완충 포장 (theme=배송/포장 리스크, count=19, npmi=0.965764)
- [sentiment/neutral] 박스 + 완충 포장 (theme=배송/포장 리스크, count=14, npmi=0.459017)
- [sentiment/neutral] 박스 + 뽁뽁이 (theme=배송/포장 리스크, count=13, npmi=0.42875)
- [sentiment/neutral] 배송 + 완충 포장 (theme=배송/포장 리스크, count=15, npmi=0.329442)
- [sentiment/neutral] 박스 + 상태 (theme=배송/포장 리스크, count=8, npmi=0.406824)
- [sentiment/neutral] 배송 + 뽁뽁이 (theme=배송/포장 리스크, count=14, npmi=0.307026)
- [sentiment/neutral] 뽁뽁이 + 포장 (theme=배송/포장 리스크, count=10, npmi=0.331651)
- [sentiment/neutral] 신경 + 카메라 만족 (theme=카메라/화질, count=6, npmi=0.405658)
- [sentiment/neutral] 카메라 만족 + 포장 (theme=배송/포장 리스크, count=13, npmi=0.292518)
- [sentiment/neutral] 박스 + 포장 (theme=배송/포장 리스크, count=12, npmi=0.295452)
- [sentiment/neutral] 도착 + 박스 (theme=배송/포장 리스크, count=5, npmi=0.418096)
- [sentiment/neutral] 완충 포장 + 포장 (theme=배송/포장 리스크, count=10, npmi=0.307683)
