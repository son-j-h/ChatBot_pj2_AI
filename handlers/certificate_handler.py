import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# ✅ 환경 변수 로드
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# ✅ 임베딩 + LLM 모델
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key,
    temperature=0.1
)

# ✅ 벡터스토어 (RAG) - 증명서 관련 문서
VECTOR_DIR = "./my_rag_db"
COLLECTION_NAME = "certificate_docs"
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_DIR,
    embedding_function=embedding_model
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type="map_reduce",
    return_source_documents=True
)

# ✅ 증명서 종류별 키워드 매핑
CERTIFICATE_KEYWORDS = {
    "수강증명서": ["수강증명서", "수강증명", "수강확인서", "수강확인", "수료확인서"],
    "참가확인서": ["참가확인서", "참가확인", "참여확인서", "참여확인"],
    "출석부": ["출석부", "출석표", "출석현황", "출석기록"],
    "수료증": ["수료증", "수료서", "완료증", "이수증"],
    "재학증명서": ["재학증명서", "재학증명", "재학확인서"],
    "성적증명서": ["성적증명서", "성적표", "성적확인서"],
    "훈련생등록": ["훈련생등록", "훈련생 등록", "등록확인서"],
    "훈련탐색표": ["훈련탐색표", "탐색표"],
    "미인정출석부": ["미인정출석부", "미인정 출석부"],
    "예비군연기서류": ["예비군연기서류", "예비군 연기", "예비군연기", "예비군 연기 서류"]
}

def identify_certificate_type(user_input: str) -> str:
    """
    사용자 입력에서 증명서 종류를 식별합니다.
    
    Args:
        user_input (str): 사용자 입력
        
    Returns:
        str: 식별된 증명서 종류 또는 "일반"
    """
    user_input_lower = user_input.lower()
    
    for cert_type, keywords in CERTIFICATE_KEYWORDS.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return cert_type
    
    return "일반"

def is_certificate_request(user_input: str) -> bool:
    """
    증명서 발급 요청 의도를 판단합니다.
    
    Args:
        user_input (str): 사용자 입력
        
    Returns:
        bool: 증명서 발급 요청 여부
    """
    request_keywords = [
        "발급", "신청", "요청", "받고싶", "필요", "출력", "뽑기", 
        "내려받기", "다운로드", "제출", "준비", "얻고싶", "받을 수 있"
    ]
    
    return any(keyword in user_input for keyword in request_keywords)

def extract_period_info(user_input: str) -> dict:
    """
    사용자 입력에서 기간 정보를 추출합니다.
    
    Args:
        user_input (str): 사용자 입력
        
    Returns:
        dict: 추출된 기간 정보
    """
    prompt = f"""
다음 문장에서 증명서 발급에 필요한 기간 정보를 JSON 형식으로 추출해 주세요.

출력 예시:
{{
  "start_date": "2025-01-01",
  "end_date": "2025-01-31",
  "has_period": true,
  "period_type": "specific"
}}

다음 항목이 포함되어야 합니다:
- start_date: 시작 날짜 (없으면 null)
- end_date: 종료 날짜 (없으면 null)
- has_period: 기간 정보가 있는지 여부 (true/false)
- period_type: "specific"(구체적 기간), "full"(전체 기간), "recent"(최근), "current"(현재) 중 하나

문장: "{user_input}"
"""
    
    try:
        response = llm.invoke(prompt).strip()
        print(f"🧠 [기간 추출 LLM 응답]:\n{response}")
        
        # JSON 코드 블록 제거 처리
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
            
        return json.loads(response)
    except Exception as e:
        print(f"[❌ 기간 추출 실패]: {e}")
        return {
            "start_date": None,
            "end_date": None,
            "has_period": False,
            "period_type": "unknown"
        }

def generate_certificate_guide(cert_type: str, user_input: str, student_info: dict = None) -> str:
    """
    증명서 종류별 맞춤형 안내를 생성합니다.
    
    Args:
        cert_type (str): 증명서 종류
        user_input (str): 사용자 입력
        student_info (dict): 학생 정보
        
    Returns:
        str: 생성된 안내 메시지
    """
    student_name = student_info.get("STUDENT_NAME", "훈련생") if student_info else "훈련생"
    
    # 기간 정보 추출
    period_info = extract_period_info(user_input)
    
    base_message = f"📋 {cert_type} 발급 안내"
    
    if cert_type in ["수강증명서", "참가확인서", "출석부"]:
        guide = f"""
{base_message}

🔸 **신청 방법**:
1. 행정문의방에 스레드 작성 (@ 행정매니저 태깅 필수)
2. 다음 양식에 따라 작성:

```
@ 행정매니저
{cert_type} 발급을 요청하는 스레드입니다.

필요한 기간: YYYY년 MM월 DD일 ~ YYYY년 MM월 DD일
필요 서류: {cert_type}
이메일: your-email@example.com
```

🔸 **발급 절차**:
"""
        
        if cert_type == "수강증명서":
            guide += """
1. 매니저 확인 후 모두싸인 전자서명으로 수강증명서 양식 발송
2. 개인정보 및 출결 내역 작성 서명 후 제출
3. 패스트캠퍼스 직인담당자 전자서명 최종 완료 후 자동 발급
"""
        elif cert_type == "참가확인서":
            guide += """
1. 매니저 확인 후 모두싸인 전자서명으로 참가확인서 발송
2. 본인 이름 서명
3. 패스트캠퍼스 직인담당자 전자서명 최종 완료 후 자동 발급
"""
        elif cert_type == "출석부":
            guide += """
1. 행정매니저 확인 후 발급
2. DM으로 발급될 수 있으나, 추가 문의는 스레드로만 진행
"""
        
        # 기간 정보가 없는 경우 안내 추가
        if not period_info.get("has_period"):
            guide += f"""

⚠️ **기간 정보 필요**: 
정확한 발급을 위해 필요한 기간을 다음과 같이 알려주세요:
예시: "2025년 1월 1일 ~ 2025년 1월 31일 {cert_type} 발급 요청합니다"
"""
        
        guide += """

🔸 **주의사항**:
- DM 문의 절대 불가 (누락 위험)
- 추가 문의는 신청한 스레드에서만 진행
- 담당 행정매니저 태깅하여 소통
"""
        
    elif cert_type in ["훈련생등록", "훈련탐색표", "미인정출석부", "예비군연기서류"]:
        guide = f"""
{base_message}

🔸 **신청 방법**:
행정문의방에 스레드 작성하여 신청

🔸 **양식**:
```
@ 행정매니저
{cert_type} 발급을 요청하는 스레드입니다.

필요 사유: [사유 작성]
이메일: your-email@example.com
```

🔸 **처리 절차**:
담당 행정매니저 확인 후 발급 진행

🔸 **주의사항**:
- 스레드를 통한 소통 원칙
- DM 문의 불가
"""
    else:
        # 일반적인 증명서 안내
        guide = f"""
📋 증명서 발급 안내

🔸 **행정문의방에서 발급 가능한 서류**:
- 수강증명서 (특정 기간 수강 내용 증명)
- 참가확인서 (전체 훈련 기간 수강 내용 증명)  
- 출석부 (출석 현황 확인)
- 국취서류 발급 (수강증명서/출석부)
- 예비군 연기 서류
- 훈련생 등록 확인
- 훈련탐색표
- 미인정출석부

🔸 **기본 신청 절차**:
1. 행정문의방 스레드 작성 (@ 행정매니저 태깅)
2. 필요 기간, 서류명, 이메일 명시
3. 담당자 확인 후 발급 진행

💡 **구체적인 서류명을 알려주시면 더 자세한 안내를 드릴 수 있습니다!**
"""
    
    return guide.strip()

def answer(user_input: str, student_id: int = None, student_info: dict = None) -> str:
    """
    증명서 발급 관련 질문에 대한 통합 처리 함수
    
    Args:
        user_input (str): 사용자 입력
        student_id (int): 학생 ID (선택사항)
        student_info (dict): 학생 정보 (선택사항)
        
    Returns:
        str: 처리 결과 메시지
    """
    if not user_input.strip():
        return "질문을 입력해주세요."
    
    try:
        # ✅ 1단계: 증명서 종류 식별
        cert_type = identify_certificate_type(user_input)
        print(f"🔍 [식별된 증명서 종류]: {cert_type}")
        
        # ✅ 2단계: 발급 요청 의도 확인
        is_request = is_certificate_request(user_input)
        
        if is_request and cert_type != "일반":
            # 구체적인 증명서 발급 요청
            print(f"📋 [증명서 발급 요청 감지]: {cert_type}")
            return generate_certificate_guide(cert_type, user_input, student_info)
        
        elif is_request and cert_type == "일반":
            # 일반적인 증명서 발급 문의
            print("📋 [일반 증명서 발급 문의]")
            return generate_certificate_guide("일반", user_input, student_info)
        
        else:
            # ✅ 3단계: 일반 정보 질문 (RAG)
            print("🔍 [일반 정보 질의 → 문서 검색]")
            
            # 증명서 관련 키워드가 포함된 경우 맞춤형 검색 쿼리 생성
            enhanced_query = user_input
            if cert_type != "일반":
                enhanced_query = f"{cert_type} {user_input}"
            
            result = qa_chain(enhanced_query)
            response = str(result["result"]).strip()
            
            # 응답이 너무 짧거나 관련성이 낮으면 기본 안내 제공
            if len(response) < 50 or "죄송" in response or "찾을 수 없" in response:
                if cert_type != "일반":
                    return generate_certificate_guide(cert_type, user_input, student_info)
                else:
                    return generate_certificate_guide("일반", user_input, student_info)
            
            # 학생 정보가 있으면 개인화된 인사 추가
            if student_info:
                student_name = student_info.get("STUDENT_NAME", "훈련생")
                response = f"{student_name}님, {response}"
            
            return response

    except Exception as e:
        print(f"[❌ 전체 처리 오류]: {e}")
        return "증명서 발급 안내 중 오류가 발생했습니다. 다시 시도해주세요."

# ✅ 테스트용 함수 (개발 시에만 사용)
def test_certificate_handler():
    """증명서 핸들러 테스트 함수"""
    test_cases = [
        "수강증명서 발급받고 싶어요",
        "출석부가 필요합니다",
        "참가확인서는 어떻게 신청하나요?",
        "증명서 발급 절차가 궁금해요",
        "1월부터 3월까지 수강증명서 발급 요청",
        "예비군 연기 서류 신청하려면?",
    ]
    
    print("🧪 [증명서 핸들러 테스트 시작]")
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i}: {test_input} ---")
        response = answer(test_input)
        print(f"응답: {response}")
    print("\n🧪 [테스트 완료]")

if __name__ == "__main__":
    # 테스트 실행
    test_certificate_handler()