import os
import json
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pymysql

# ✅ 환경 변수 로드
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise EnvironmentError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# ✅ 임베딩 + LLM 모델
embedding_model = OpenAIEmbeddings(
    openai_api_key=openai_key,
    model="text-embedding-3-small"
)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    openai_api_key=openai_key
)

# ✅ 벡터스토어 (RAG)
VECTOR_DIR = "./my_rag_db"
COLLECTION_NAME = "admin_docs"
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

# ✅ 의도 판단 키워드
INTENT_KEYWORDS = [
    "쓰고 싶", "신청", "할래", "싶어", "내고 싶",
    "아프", "조퇴", "쉬고 싶", "쉴래", "조퇴할래", "병원", "몸이 안 좋아"
]

def is_leave_intent_rule(text: str) -> bool:
    return any(keyword in text for keyword in INTENT_KEYWORDS)

def is_leave_intent_llm(text: str) -> bool:
    prompt = f"""
다음 문장이 '휴가', '공가', '조퇴', '병가' 등을 신청하려는 의도인지 판단해 주세요.
- 문장이 그런 의도라면 "예", 아니라면 "아니오"로만 답변하세요.
문장: "{text}"
"""
    try:
        response = llm.predict(prompt).strip().lower()
        return "예" in response
    except Exception as e:
        print(f"[❌ LLM 판단 오류]: {e}")
        return False

def is_leave_intent(text: str) -> bool:
    return is_leave_intent_rule(text) or is_leave_intent_llm(text)

# ✅ LLM으로 날짜, 사유 파싱 (JSON 안전 파싱)
def extract_leave_info(user_input: str) -> dict:
    prompt = f"""
다음 문장에서 휴가 신청 정보를 JSON 형식으로 추출해주세요.
문장: "{user_input}"

예시:
{{
  "start_date": "2025-08-01",
  "end_date": "2025-08-03",
  "reason": "병원 진료"
}}

없는 항목은 null 로 작성해주세요.
"""
    try:
        response = llm.predict(prompt).strip()
        print(f"🧠 [LLM 파싱 응답]:\n{response}")
        return json.loads(response)
    except Exception as e:
        print(f"[❌ LLM 파싱 실패]: {e}")
        return {
            "start_date": None,
            "end_date": None,
            "reason": None
        }

# ✅ DB insert
def insert_attendance_request(
    student_id: int,
    type_big: str,
    type_small: str,
    start_dt: str,
    end_dt: str,
    reason: str
) -> bool:
    try:
        db_port = int(os.getenv("DB_PORT", 3306))  # ✅ 포트 환경변수 처리

        conn = pymysql.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=db_port,
            user=os.getenv("DB_USER", "user"),
            password=os.getenv("DB_PASSWORD", "password"),
            db=os.getenv("DB_NAME", "bootcamp"),
            charset="utf8mb4",
            autocommit=True
        )
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO ATTENDANCE_REQUESTS
                (STUDENT_ID, TYPE_BIG, TYPE_SMALL, START_DATETIME, END_DATETIME, REASON)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                student_id, type_big, type_small, start_dt, end_dt, reason
            ))
        conn.close()
        return True
    except Exception as e:
        print(f"[❌ DB insert 오류]: {e}")
        return False

# ✅ 메인 응답 함수
def answer(user_input: str, student_id: int = 1) -> str:
    if not user_input.strip():
        return "질문을 입력해주세요."

    try:
        if is_leave_intent(user_input):
            print("🧭 [휴가 신청 의도 판단됨 → LLM 파싱 시도]")
            info = extract_leave_info(user_input)
            start = info.get("start_date")
            end = info.get("end_date")
            reason = info.get("reason")

            if not (start and end and reason):
                return (
                    "휴가 또는 조퇴를 신청하시려는 것 같네요!\n"
                    "📅 언제부터 언제까지 쉬실 예정인가요?\n"
                    "📝 그리고 사유도 함께 알려주세요!"
                )

            success = insert_attendance_request(
                student_id=student_id,
                type_big="휴가",
                type_small="기타",
                start_dt=start,
                end_dt=end,
                reason=reason
            )

            if success:
                return (
                    f"✅ 휴가 신청이 정상적으로 접수되었습니다!\n"
                    f"⏰ 기간: {start} ~ {end}\n"
                    f"📝 사유: {reason}\n"
                    f"승인까지 잠시 기다려주세요."
                )
            else:
                return "❌ 휴가 신청 처리 중 오류가 발생했습니다. 다시 시도해주세요."

        # 일반 정보 질의 → RAG
        print("🔍 [일반 정보 질의 → 문서 검색 시작]")
        result = qa_chain(user_input)

        source_docs = result["source_documents"]
        print(f"\n📚 [참고한 문서 수]: {len(source_docs)}")
        for i, doc in enumerate(source_docs):
            print(f"\n📄 문서 {i+1}:\n{doc.page_content[:300]}")

        return str(result["result"])

    except Exception as e:
        print(f"[❌ 전체 처리 오류]: {e}")
        return "답변 중 오류가 발생했습니다. 다시 시도해주세요."
