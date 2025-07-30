import os
from flask import jsonify
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import datetime
import re # 정규 표현식 모듈 임포트

# 환경변수 로드
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# 벡터 DB 저장 위치 및 설정
PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../my_rag_db"))
COLLECTION_NAME = "admin_docs"

# 현재 날짜와 시간 정보
CURRENT_DATE = datetime.datetime.now().strftime("%Y년 %m월 %d일")
CURRENT_TIME = datetime.datetime.now().strftime("%H시 %M분")

# --- 기존 코드의 1, 2, 3번 함수는 LLM 경쟁을 위해 약간 수정되거나 그대로 사용됩니다 ---

# ✅ 1. 벡터 DB 로딩 (임베딩 모델은 GoogleGenerativeAIEmbeddings로 통일)
def load_vectorstore():
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    if not os.path.exists(PERSIST_DIR):
        raise ValueError("❌ 벡터 DB 폴더가 존재하지 않습니다. 먼저 생성해 주세요.")


    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})

# ✅ 2. 프롬프트 템플릿 정의 (현재 날짜/시간 플레이스홀더 추가)
def get_subsidy_prompt():
    system_template = """너는 패스트캠퍼스의 훈련장려금 전문 상담 챗봇이야.
사용자의 질문에 대해 아래 참고 문서 내용만 기반으로 정확하고 친절하게 답변해.
현재 날짜는 {current_date} {current_time} 입니다.

- 참고 문서에 없는 정보는 "자료에 없음"이라고 말해.
- 핵심 정보를 간결하고 쉽게 설명해 줘.
- 필요한 경우 bullet list 형식으로 정리해 줘.
- 문서 내용을 직접 인용해도 좋아.

참고 문서:
{context}
"""
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{question}")
    ])

# ✅ 3. LCEL 체인 구성 함수 (LLM 모델을 인자로 받아 체인 생성)
def build_llm_chain(llm_model):
    retriever = load_vectorstore()
    prompt = get_subsidy_prompt()

    chain = (
        {
            "context": lambda x: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["question"])]),
            "question": lambda x: x["question"],
            "current_date": lambda x: CURRENT_DATE, # 프롬프트에 현재 날짜 전달
            "current_time": lambda x: CURRENT_TIME  # 프롬프트에 현재 시간 전달
        }
        | prompt
        | llm_model
        | StrOutputParser()
    )
    return chain

# ✅ 4. 답변 평가 함수 (가장 핵심적인 로직)
def evaluate_answers(question: str, answers: dict) -> str:
    """
    여러 LLM의 답변을 평가하여 최적의 답변을 선정합니다.
    다양한 기준을 통해 점수를 부여하고, 가장 높은 점수를 받은 답변을 반환합니다.
    """
    best_answer = None
    best_score = -float('inf') # 초기 점수를 매우 낮게 설정

    print("\n" + "="*50)
    print("✨ 답변 평가 시작 ✨")
    print("="*50)

    # 평가 기준에 사용될 키워드 및 패턴 정의 (소문자 변환 후 비교)
    uncertainty_keywords = ["잘 모르겠습니다", "자료에 없음", "정보가 제한적", "정확히 알 수 없습니다",
                            "파악하기 어렵습니다", "알려드릴 수 없습니다", "죄송합니다", "이해하지 못했습니다"]
    
    # 질문에서 핵심 키워드 추출 (간단한 방법, 필요시 KoNLPy 등 NLP 라이브러리 사용 가능)
    question_keywords = [
        word for word in question.replace('?', '').replace('.', '').replace(',', '').split()
        if len(word) > 1 and word.lower() not in ["은", "는", "이", "가", "을", "를", "에", "에서", "와", "과",
                                                 "어떻게", "되나요", "무엇인가요", "언제", "어디서", "무엇을", "누가", "대한", "관한"]
    ]
    # '훈련장려금'은 항상 중요한 핵심 키워드이므로 명시적으로 추가
    if '훈련장려금' not in [k.lower() for k in question_keywords]:
        question_keywords.append('훈련장려금')

    # 날짜/시간 및 최신 정보 관련 키워드
    time_related_keywords = ["최신", "현재", "최근", "오늘", "지금",
                              str(datetime.datetime.now().year), str(datetime.datetime.now().month)] # 예: 2024, 7 등

    for model_name, answer_info in answers.items():
        answer_text = answer_info.get("answer", "")
        error = answer_info.get("error")
        score = 0
        answer_lower = answer_text.lower() # 소문자로 변환하여 비교 용이

        print(f"\n--- 평가 중: [{model_name}] ---")

        # 1. 오류가 없는 답변에 높은 점수 부여 (기본 점수 및 필수 조건)
        if error:
            score -= 500 # 오류가 있으면 거의 선택되지 않도록 매우 낮은 점수 부여
            continue # 오류가 있는 답변은 더 이상 평가하지 않고 다음 모델로 넘어감
        else:
            score += 100 # 오류 없음은 기본 가점

        # 2. 불확실성/부정확성 표현 감점
        for keyword in uncertainty_keywords:
            if keyword in answer_lower:
                score -= 70 # 불확실성 키워드 발견 시 큰 감점
                break # 하나만 발견되어도 해당 감점 적용 후 다음 평가로
        
        # 3. 핵심 키워드 포함 여부 가점
        found_keywords_count = 0
        for keyword in question_keywords:
            if keyword.lower() in answer_lower:
                found_keywords_count += 1
        score += (found_keywords_count * 10) # 키워드 1개당 10점 가점

        # 4. 불릿 리스트 형식 포함 시 가점 (가독성 향상)
        if re.search(r'[-*•]\s|^\d+\.\s', answer_text, re.MULTILINE): # 불릿 또는 숫자 목록 패턴 확인
            score += 20 # 불릿 리스트 형태 발견 시 가점

        # 5. 답변 길이에 따른 점수 부여 (적절한 길이 선호)
        answer_length = len(answer_text)
        if 50 < answer_length <= 300: # 50자 초과 300자 이하일 때 가점 (핵심 정보 + 간결성)
            score += (answer_length // 20) # 20자 당 1점 가산 (최대 15점)
        elif answer_length <= 50: # 너무 짧은 답변 감점
            score -= 20
        elif answer_length > 300: # 너무 긴 답변 감점 (요약 능력 부족으로 간주, 감점 폭은 짧은 것보다 작게)
            score -= 10

        # 6. 날짜/시간 또는 최신 정보 언급 시 가점 (프롬프트에 현재 날짜를 주므로, 이를 반영했는지 확인)
        current_year = str(datetime.datetime.now().year)
        current_month = str(datetime.datetime.now().month)
        
        for keyword in time_related_keywords:
            if keyword.lower() in answer_lower or current_year in answer_text or current_month in answer_text:
                score += 15
                break # 하나만 발견되어도 가점하고 다음으로

        # 7. 구체적인 숫자/데이터 포함 시 가점 (예: 금액, 기간 등)
        if re.search(r'\d{1,3}(,\d{3})*(\.\d+)?', answer_text): # 숫자 (천단위 구분자, 소수점 포함) 패턴 확인
            score += 15

        print(f"[{model_name}] 최종 점수: {score}")

        # 최고 점수 답변 갱신
        if score > best_score:
            best_score = score
            best_answer = answer_text
        elif score == best_score and best_answer is None: # 첫 번째 동점일 경우
            best_answer = answer_text
        # 동점일 경우 특정 모델 우선 순위를 정하고 싶다면 여기에 로직 추가

    print("\n" + "="*50)
    print(f"🏆 최종 선정 답변 점수: {best_score}")
    print("="*50)

    if best_answer is None or best_answer.strip() == "":
        return "죄송합니다. 현재 질문에 대한 적절한 답변을 찾을 수 없습니다."
    
    return best_answer

# ✅ 5. answer() 함수 (LLM 경쟁 로직 포함)
def answer(question: str) -> str: # 비동기 제거
    if not question.strip():
        return jsonify({"answer": "질문을 입력해주세요."}), 200

    # 두 가지 Gemini LLM 모델 인스턴스 생성
    llm_gemini_flash = GoogleGenerativeAI(
        model="gemini-2.5-flash-lite", # 첫 번째 경쟁 모델
        google_api_key=google_api_key,
        temperature=0.2,
        max_output_tokens=800
    )
    llm_gemini_pro = GoogleGenerativeAI(
        model="gemini-2.5-pro", # 두 번째 경쟁 모델
        google_api_key=google_api_key,
        temperature=0.2,
        max_output_tokens=800
    )

    # 각 LLM 별 체인 생성
    _chain_gemini_flash = build_llm_chain(llm_gemini_flash)
    _chain_gemini_pro = build_llm_chain(llm_gemini_pro)

    results = {}

    # Gemini Flash 답변 생성
    try:
        # invoke 메서드는 동기적으로 작동합니다.
        flash_answer = _chain_gemini_flash.invoke({"question": question}) 
        results[llm_gemini_flash.model] = {"answer": flash_answer}
    except Exception as e:
        print(f"[❌ {llm_gemini_flash.model} 오류 발생]: {e}")
        results[llm_gemini_flash.model] = {"error": str(e)}

    # Gemini Pro 답변 생성
    try:
        # invoke 메서드는 동기적으로 작동합니다.
        pro_answer = _chain_gemini_pro.invoke({"question": question})
        results[llm_gemini_pro.model] = {"answer": pro_answer}
    except Exception as e:
        print(f"[❌ {llm_gemini_pro.model} 오류 발생]: {e}")
        results[llm_gemini_pro.model] = {"error": str(e)}

    # 취합된 결과를 바탕으로 최적 답변 선택 및 반환
    response_text = evaluate_answers(question, results)
    return response_text;