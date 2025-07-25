import os
import json
from datetime import datetime
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
COLLECTION_NAME = "leave_docs"
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
다음 문장에서 조퇴/휴가 신청 정보를 JSON 형식으로 추출해 주세요.

출력 예시는 다음과 같습니다:
{{
  "start_date": "2025-08-01",
  "end_date": "2025-08-01",
  "start_time": "14:00",   # 시간 정보는 조퇴일 경우만 사용, 없으면 null
  "end_time": "18:00",     # 시간 정보는 조퇴일 경우만 사용, 없으면 null
  "reason": "두통 때문에 병원 방문",
  "type_big": "조퇴",
  "type_small": "두통"
}}

다음 항목이 반드시 포함되어야 합니다:
- start_date, end_date (날짜가 없으면 null)
- start_time, end_time (조퇴일 경우만, 없으면 null)
- reason (문장에서의 전체 사유)
- type_big ("휴가", "병가", "공가", "조퇴" 중 하나)
- type_small (사유 요약 10자 이내)

문장: "{user_input}"
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
            "start_time": None,
            "end_time": None,
            "reason": None,
            "type_big": None,
            "type_small": None
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
        db_port = int(os.getenv("MYSQL_PORT", 3306))  # ✅ 포트 환경변수 처리

        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=db_port,
            user=os.getenv("MYSQL_USER", "user"),
            password=os.getenv("MYSQL_PASSWORD", "password"),
            db=os.getenv("MYSQL_DB", "bootcamp"),
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

def detect_leave_type(text: str) -> str:
    text = text.lower()
    if "병가" in text:
        return "병가"
    elif "공가" in text:
        return "공가"
    elif "조퇴" in text:
        return "조퇴"
    else:
        return "휴가"  # 기본값

# ✅ 메인 응답 함수

def answer(question: str) -> str:
    if not question.strip():
        return "질문을 입력해주세요."

    try:
        if is_leave_intent(question):
            print("🧭 [휴가 신청 의도 판단됨 → LLM 파싱 시도]")
            info = extract_leave_info(question)

            start = info.get("start_date")
            end = info.get("end_date")
            start_time = info.get("start_time")
            end_time = info.get("end_time")
            reason = info.get("reason")
            type_big = info.get("type_big")
            type_small = info.get("type_small")

            if type_big == "조퇴":
                today = datetime.today().strftime("%Y-%m-%d")
                start = today
                end = today

            is_complete = all([start, end, reason])
            if type_big == "조퇴":
                is_complete = is_complete and start_time and end_time

            if not is_complete:
                type_label = type_big if type_big else "휴가 또는 조퇴 등"
                return (
                    f"{type_label}를 신청하시려는 것 같네요!\n"
                    f"📅 언제부터 언제까지 예정인가요? 오늘인가요?\n"
                    f"📝 그리고 사유도 함께 알려주세요!"
                )

            # ⛔ student_id는 없으므로 강제로 1로 고정 (정상 처리 X)
            success = insert_attendance_request(
                student_id=1,
                type_big=type_big or "휴가",
                type_small=type_small or "기타",
                start_dt=start,
                end_dt=end,
                reason=reason
            )

            if success:
                msg = (
                    f"✅ {type_big} 신청이 정상적으로 접수되었습니다!\n"
                    f"⏰ 기간: {start} ~ {end}\n"
                )
                if type_big == "조퇴":
                    msg += f"🕒 시간: {start_time} ~ {end_time}\n"
                msg += f"📝 사유: {reason}\n승인까지 잠시 기다려주세요."
                return msg
            else:
                return "❌ 신청 처리 중 오류가 발생했습니다. 다시 시도해주세요."

        # 일반 정보 질의 (RAG)
        print("🔍 [일반 정보 질의 → 문서 검색 시작]")
        result = qa_chain(question)

        source_docs = result["source_documents"]
        print(f"\n📚 [참고한 문서 수]: {len(source_docs)}")
        for i, doc in enumerate(source_docs):
            print(f"\n📄 문서 {i+1}:\n{doc.page_content[:300]}")

        return str(result["result"])

    except Exception as e:
        print(f"[❌ 전체 처리 오류]: {e}")
        return "답변 중 오류가 발생했습니다. 다시 시도해주세요."