import os
import json
import re
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
  "start_time": "14:00",
  "end_time": "18:00",
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
def insert_attendance_request(student_id, type_big, type_small, start_dt, end_dt, reason) -> bool:
    try:
        db_port = int(os.getenv("MYSQL_PORT", 3306))
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

# ✅ DB 조회 (최근 신청 내역)
def get_attendance_records(student_id: int) -> list:
    try:
        db_port = int(os.getenv("MYSQL_PORT", 3306))
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=db_port,
            user=os.getenv("MYSQL_USER", "user"),
            password=os.getenv("MYSQL_PASSWORD", "password"),
            db=os.getenv("MYSQL_DB", "bootcamp"),
            charset="utf8mb4"
        )
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT * FROM ATTENDANCE_REQUESTS
                WHERE STUDENT_ID = %s
                ORDER BY REQUEST_AT DESC
                LIMIT 10
            """
            cursor.execute(sql, (student_id,))
            result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"[❌ DB 조회 오류]: {e}")
        return []

# ✅ '대기중' 내역만 조회
def get_pending_attendance_requests(student_id: int) -> list:
    """
    특정 학생의 대기중 상태인 휴가/병가/공가/조퇴 신청 내역을 조회합니다.
    
    Args:
        student_id (int): 학생 ID
        
    Returns:
        list: 대기중 신청 내역 리스트 (각 항목은 딕셔너리)
    """
    try:
        db_port = int(os.getenv("MYSQL_PORT", 3306))
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=db_port,
            user=os.getenv("MYSQL_USER", "user"),
            password=os.getenv("MYSQL_PASSWORD", "password"),
            db=os.getenv("MYSQL_DB", "bootcamp"),
            charset="utf8mb4"
        )
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT REQUEST_ID, TYPE_BIG, TYPE_SMALL, START_DATETIME, END_DATETIME, 
                       REASON, STATUS, REQUEST_AT
                FROM ATTENDANCE_REQUESTS
                WHERE STUDENT_ID = %s AND STATUS = '대기중'
                AND TYPE_BIG IN ('휴가', '병가', '공가', '조퇴')
                ORDER BY REQUEST_AT DESC
            """
            cursor.execute(sql, (student_id,))
            result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"[❌ 대기중 내역 조회 오류]: {e}")
        return []

# ✅ 취소 처리
def cancel_attendance_request(request_id: int) -> bool:
    """
    특정 신청 ID의 상태를 '취소됨'으로 변경합니다.
    
    Args:
        request_id (int): 취소할 신청의 REQUEST_ID
        
    Returns:
        bool: 취소 성공 여부
    """
    try:
        db_port = int(os.getenv("MYSQL_PORT", 3306))
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
                UPDATE ATTENDANCE_REQUESTS
                SET STATUS = '취소됨'
                WHERE REQUEST_ID = %s AND STATUS = '대기중'
            """
            affected = cursor.execute(sql, (request_id,))
        conn.close()
        return affected > 0
    except Exception as e:
        print(f"[❌ 신청 취소 오류]: {e}")
        return False

# ✅ 취소 대상 식별을 위한 LLM 함수
def identify_cancel_target(user_input: str, pending_requests: list) -> dict:
    """
    사용자의 자연어 입력에서 취소하고자 하는 신청을 식별합니다.
    
    Args:
        user_input (str): 사용자 입력
        pending_requests (list): 대기중 신청 내역 리스트
        
    Returns:
        dict: {"request_id": int 또는 None, "reason": str}
    """
    if not pending_requests:
        return {"request_id": None, "reason": "취소 가능한 신청이 없습니다."}
    
    # 명시적인 ID 패턴 확인 (ID:123, REQUEST_ID:123 등)
    id_patterns = [
        r"(?:ID|REQUEST_ID)[:\s]*(\d+)",
        r"(\d+)번",
        r"신청\s*(\d+)",
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            request_id = int(match.group(1))
            # 해당 ID가 실제 대기중 신청에 있는지 확인
            if any(req['REQUEST_ID'] == request_id for req in pending_requests):
                return {"request_id": request_id, "reason": f"신청 ID {request_id}번을 취소 대상으로 식별했습니다."}
            else:
                return {"request_id": None, "reason": f"신청 ID {request_id}번은 취소 가능한 대기중 상태가 아닙니다."}
    
    # LLM을 사용한 자연어 기반 식별
    if len(pending_requests) == 1:
        # 대기중 신청이 하나뿐이면 자동으로 그것을 선택
        return {
            "request_id": pending_requests[0]['REQUEST_ID'], 
            "reason": "대기중인 신청이 하나뿐이므로 해당 신청을 취소 대상으로 선택했습니다."
        }
    
    # 여러 신청이 있는 경우 LLM으로 매칭 시도
    requests_info = ""
    for i, req in enumerate(pending_requests, 1):
        requests_info += f"{i}. ID:{req['REQUEST_ID']} - {req['TYPE_BIG']} ({req['START_DATETIME']} ~ {req['END_DATETIME']}) - {req['REASON']}\n"
    
    prompt = f"""
사용자가 다음과 같이 말했습니다: "{user_input}"

현재 취소 가능한 신청 목록:
{requests_info}

사용자가 취소하고 싶어하는 신청의 REQUEST_ID를 숫자로만 답변해주세요.
명확하게 특정할 수 없다면 "불명확"이라고 답변해주세요.
"""
    
    try:
        response = llm.predict(prompt).strip()
        if response.isdigit():
            request_id = int(response)
            if any(req['REQUEST_ID'] == request_id for req in pending_requests):
                return {"request_id": request_id, "reason": f"자연어 분석을 통해 신청 ID {request_id}번을 취소 대상으로 식별했습니다."}
        
        return {"request_id": None, "reason": "취소하고자 하는 구체적인 신청을 특정할 수 없습니다. 'ID:번호 취소' 형식으로 말씀해주세요."}
    except Exception as e:
        print(f"[❌ LLM 취소 대상 식별 오류]: {e}")
        return {"request_id": None, "reason": "취소 대상 식별 중 오류가 발생했습니다."}

# ✅ 메인 응답 핸들러
def answer(user_input: str, student_id: int = None, student_info: dict = None) -> str:
    """
    휴가/병가/공가/조퇴 관련 질문에 대한 통합 처리 함수
    
    Args:
        user_input (str): 사용자 입력
        student_id (int): 학생 ID
        student_info (dict): 학생 정보 (선택사항)
        
    Returns:
        str: 처리 결과 메시지
    """
    if not user_input.strip():
        return "질문을 입력해주세요."
    if student_id is None:
        student_id = 1

    try:
        # ✅ 1단계: 취소 요청 처리
        if "취소" in user_input:
            print("🚫 [취소 의도 감지됨]")
            
            # 직접적인 ID 기반 취소 (ID:123 취소)
            id_match = re.search(r"(?:ID|REQUEST_ID)[:\s]*(\d+)", user_input, re.IGNORECASE)
            if id_match:
                request_id = int(id_match.group(1))
                success = cancel_attendance_request(request_id)
                if success:
                    return f"✅ 신청 ID {request_id}번이 성공적으로 취소되었습니다."
                else:
                    return f"❌ 신청 ID {request_id}번은 취소할 수 없거나 이미 처리된 상태입니다."
            
            # 자연어 기반 취소 처리
            pending_requests = get_pending_attendance_requests(student_id)
            if not pending_requests:
                return "취소 가능한 대기중 상태의 신청 내역이 없습니다."
            
            # 취소 대상 식별
            cancel_result = identify_cancel_target(user_input, pending_requests)
            
            if cancel_result["request_id"]:
                # 취소 실행
                success = cancel_attendance_request(cancel_result["request_id"])
                if success:
                    return f"✅ 신청 ID {cancel_result['request_id']}번이 성공적으로 취소되었습니다."
                else:
                    return f"❌ 신청 ID {cancel_result['request_id']}번 취소 처리 중 오류가 발생했습니다."
            else:
                # 취소 가능한 목록 표시
                response = "🛑 취소 가능한 신청 내역:\n"
                for i, req in enumerate(pending_requests, 1):
                    response += (
                        f"\n🔸 신청 {i}번 (ID: {req['REQUEST_ID']})\n"
                        f"  📅 {req['START_DATETIME']} ~ {req['END_DATETIME']}\n"
                        f"  📝 사유: {req['REASON']}\n"
                        f"  📌 유형: {req['TYPE_BIG']} / {req['TYPE_SMALL']}\n"
                        f"  📊 상태: {req['STATUS']}"
                    )
                response += f"\n\n{cancel_result['reason']}"
                response += "\n취소하려면 'ID:숫자 취소'라고 말해주세요. 예: ID:123 취소"
                return response

        # ✅ 2단계: 일반 신청 조회
        if any(k in user_input for k in ["내역", "조회", "신청한", "상태", "확인"]):
            print("🔎 [조회 의도 감지됨]")
            records = get_attendance_records(student_id)

            # 유형 필터
            filter_type = None
            for t in ["휴가", "병가", "공가", "조퇴"]:
                if t in user_input:
                    filter_type = t
                    break
            if filter_type:
                records = [r for r in records if r["TYPE_BIG"] == filter_type]

            if not records:
                return "최근 신청 내역이 없습니다."

            response = "📋 최근 신청 내역\n"
            for i, r in enumerate(records, 1):
                response += (
                    f"\n🔹 신청 {i}번 (ID: {r.get('REQUEST_ID', 'N/A')})\n"
                    f"  📅 {r['START_DATETIME']} ~ {r['END_DATETIME']}\n"
                    f"  📝 사유: {r['REASON']}\n"
                    f"  📌 유형: {r['TYPE_BIG']} / {r['TYPE_SMALL']}\n"
                    f"  📊 상태: {r['STATUS']}\n"
                )
            return response.strip()

        # ✅ 3단계: 신청 처리
        if is_leave_intent(user_input):
            print("🧭 [휴가 신청 의도 판단됨 → LLM 파싱 시도]")
            info = extract_leave_info(user_input)
            start = info.get("start_date")
            end = info.get("end_date")
            reason = info.get("reason")
            type_big = info.get("type_big") or "휴가"
            type_small = info.get("type_small") or "기타"

            if not (start and end and reason):
                return (
                    f"{type_big}를 신청하시려는 것 같네요!\n"
                    "📅 언제부터 언제까지 예정인가요?\n"
                    "📝 그리고 사유도 함께 알려주세요!"
                )

            success = insert_attendance_request(
                student_id=student_id,
                type_big=type_big,
                type_small=type_small,
                start_dt=start,
                end_dt=end,
                reason=reason
            )

            if success:
                return (
                    f"✅ {type_big} 신청이 정상적으로 접수되었습니다!\n"
                    f"⏰ 기간: {start} ~ {end}\n"
                    f"📝 사유: {reason}\n"
                    f"승인까지 잠시 기다려주세요."
                )
            else:
                return "❌ 신청 처리 중 오류가 발생했습니다. 다시 시도해주세요."

        # ✅ 4단계: 일반 정보 질문 (RAG)
        print("🔍 [일반 정보 질의 → 문서 검색]")
        result = qa_chain(user_input)
        return str(result["result"])

    except Exception as e:
        print(f"[❌ 전체 처리 오류]: {e}")
        return "답변 중 오류가 발생했습니다. 다시 시도해주세요."