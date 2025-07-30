import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import pymysql

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
    temperature=0.2
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
    "아프", "조퇴", "쉬고 싶", "쉴래", "조퇴할래", "병원", "몸이 안 좋어"
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
        response = llm.invoke(prompt).strip().lower()
        return "예" in response
    except Exception as e:
        print(f"[❌ LLM 판단 오류]: {e}")
        return False

def is_leave_intent(text: str) -> bool:
    return is_leave_intent_rule(text) or is_leave_intent_llm(text)

# ✅ 조회 의도 판단 함수 (신규 추가)
def is_inquiry_intent(text: str) -> bool:
    """
    조회 의도를 명확히 판단하는 함수
    신청 의도와 겹치지 않도록 더 엄격한 조건 적용
    """
    inquiry_keywords = ["내역", "조회", "보여줘", "확인", "목록", "상태", "신청한"]
    # 신청 의도 키워드가 포함되어 있으면 조회 의도로 보지 않음
    apply_keywords = ["신청할", "할래", "하고 싶", "내고 싶", "쓰고 싶"]
    
    has_inquiry = any(keyword in text for keyword in inquiry_keywords)
    has_apply = any(keyword in text for keyword in apply_keywords)
    
    # 신청 의도가 명확하면 조회로 보지 않음
    if has_apply:
        return False
    
    return has_inquiry

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
        response = llm.invoke(prompt).strip()
        print(f"🧠 [LLM 파싱 응답]:\n{response}")
        # JSON 코드 블록 제거 처리
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
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

# ✅ DB 조회 (최근 신청 내역) - limit 매개변수 추가
def get_attendance_records(student_id: int, limit: int = 10) -> list:
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
                LIMIT %s
            """
            cursor.execute(sql, (student_id, limit))
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

# ✅ 특정 유형의 '대기중' 내역만 조회 (취소용)
def get_pending_requests_by_type(student_id: int, type_big: str) -> list:
    """
    특정 학생의 특정 유형(휴가/병가/공가/조퇴)의 대기중 상태 신청 내역을 조회합니다.
    (취소 기능에서 사용)
    
    Args:
        student_id (int): 학생 ID
        type_big (str): 신청 유형 ('휴가', '병가', '공가', '조퇴')
        
    Returns:
        list: 해당 유형의 대기중 신청 내역 리스트 (각 항목은 딕셔너리)
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
                WHERE STUDENT_ID = %s AND STATUS = '대기중' AND TYPE_BIG = %s
                ORDER BY REQUEST_AT DESC
            """
            cursor.execute(sql, (student_id, type_big))
            result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"[❌ 특정 유형 대기중 내역 조회 오류]: {e}")
        return []

# ✅ 특정 유형의 전체 내역 조회 (상태값 제한 없음)
def get_attendance_records_by_type(student_id: int, type_big: str) -> list:
    """
    특정 학생의 특정 유형(휴가/병가/공가/조퇴)의 모든 상태 신청 내역을 조회합니다.
    
    Args:
        student_id (int): 학생 ID
        type_big (str): 신청 유형 ('휴가', '병가', '공가', '조퇴')
        
    Returns:
        list: 해당 유형의 모든 신청 내역 리스트 (각 항목은 딕셔너리)
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
                WHERE STUDENT_ID = %s AND TYPE_BIG = %s
                ORDER BY REQUEST_AT DESC
                LIMIT 20
            """
            cursor.execute(sql, (student_id, type_big))
            result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"[❌ 특정 유형 내역 조회 오류]: {e}")
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

# ✅ 취소 대상 식별을 위한 함수 (자동 취소 제거, 목록 표시로 변경)
def identify_cancel_target(user_input: str, pending_requests: list) -> dict:
    """
    사용자의 자연어 입력에서 취소하고자 하는 신청을 식별합니다.
    명확한 ID 지정이 없으면 목록만 표시하고 자동 취소하지 않습니다.
    
    Args:
        user_input (str): 사용자 입력
        pending_requests (list): 대기중 신청 내역 리스트
        
    Returns:
        dict: {"request_id": int 또는 None, "reason": str, "show_list": bool}
    """
    if not pending_requests:
        return {"request_id": None, "reason": "취소 가능한 신청이 없습니다.", "show_list": False}
    
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
                return {
                    "request_id": request_id, 
                    "reason": f"신청 ID {request_id}번을 취소 대상으로 식별했습니다.",
                    "show_list": False
                }
            else:
                return {
                    "request_id": None, 
                    "reason": f"신청 ID {request_id}번은 취소 가능한 대기중 상태가 아닙니다.",
                    "show_list": True
                }
    
    # 명확한 ID 지정이 없으면 목록만 표시 (자동 취소 제거)
    return {
        "request_id": None, 
        "reason": "취소하고자 하는 구체적인 신청을 특정할 수 없습니다.",
        "show_list": True
    }

# ✅ 간결한 출결 내역 표시 함수 (신규 추가)
def format_brief_attendance_records(records: list, title: str = "📋 최근 출결 신청 내역") -> str:
    """
    출결 신청 내역을 간결하게 포맷팅합니다.
    
    Args:
        records (list): 신청 내역 리스트
        title (str): 표시할 제목
        
    Returns:
        str: 포맷팅된 문자열
    """
    if not records:
        return "출결 신청 내역이 없습니다."
    
    response = f"{title} (최근 5건)\n"
    for i, record in enumerate(records[:5], 1):  # 상위 5개만 표시
        # 날짜 포맷팅 (시간 부분 제거)
        start_date = str(record['START_DATETIME']).split(' ')[0] if record['START_DATETIME'] else 'N/A'
        end_date = str(record['END_DATETIME']).split(' ')[0] if record['END_DATETIME'] else 'N/A'
        
        # 사유 간략화 (30자 제한)
        reason = record['REASON'][:30] + '...' if len(record['REASON']) > 30 else record['REASON']
        
        response += (
            f"\n{i}. 📅 {start_date}~{end_date} | "
            f"📌 {record['TYPE_BIG']} | "
            f"📊 {record['STATUS']} | "
            f"📝 {reason}"
        )
    
    if len(records) > 5:
        response += f"\n\n... 외 {len(records) - 5}건 더 있습니다."
    
    return response.strip()

# ✅ 메인 응답 핸들러 (우선순위 및 로직 개선)
def answer(user_input: str, student_id: int = None, student_info: dict = None) -> str:
    """
    휴가/병가/공가/조퇴 관련 질문에 대한 통합 처리 함수
    개선된 의도 분기 우선순위:
    1. 취소 요청 처리
    2. 신청 의도 처리
    3. 조회 의도 처리
    4. 일반 정보 질문 (RAG)
    
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
        # ✅ 1단계: 취소 요청 처리 (최우선)
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
            
            # 취소 대상 신청 내역 조회 (유형별 또는 전체)
            pending_requests = []
            target_type = None
            
            # 전체 출결 취소 요청
            if "출결" in user_input:
                print("🔍 [전체 출결 취소 요청 감지]")
                pending_requests = get_pending_attendance_requests(student_id)
                target_type = "출결"
            else:
                # 유형별 취소 요청
                for leave_type in ["휴가", "병가", "공가", "조퇴"]:
                    if leave_type in user_input:
                        print(f"🔍 [{leave_type} 취소 요청 감지]")
                        pending_requests = get_pending_requests_by_type(student_id, leave_type)
                        target_type = leave_type
                        break
                
                # 유형이 명시되지 않았다면 전체 조회
                if not pending_requests and not target_type:
                    print("🔍 [일반 취소 요청 - 전체 조회]")
                    pending_requests = get_pending_attendance_requests(student_id)
                    target_type = "전체"
            
            # 취소 가능한 신청이 없는 경우
            if not pending_requests:
                if target_type and target_type != "전체":
                    return f"취소 가능한 대기중 상태의 {target_type} 신청 내역이 없습니다."
                else:
                    return "취소 가능한 대기중 상태의 신청 내역이 없습니다."
            
            # 취소 대상 식별 (자동 취소 제거됨)
            cancel_result = identify_cancel_target(user_input, pending_requests)
            
            if cancel_result["request_id"]:
                # 명확한 ID가 있을 때만 취소 실행
                success = cancel_attendance_request(cancel_result["request_id"])
                if success:
                    return f"✅ 신청 ID {cancel_result['request_id']}번이 성공적으로 취소되었습니다."
                else:
                    return f"❌ 신청 ID {cancel_result['request_id']}번 취소 처리 중 오류가 발생했습니다."
            else:
                # 목록 표시 및 선택 유도
                type_display = f" ({target_type})" if target_type and target_type != "전체" else ""
                response = f"🛑 취소 가능한 신청 내역{type_display}:\n"
                for i, req in enumerate(pending_requests, 1):
                    start_date = str(req['START_DATETIME']).split(' ')[0] if req['START_DATETIME'] else 'N/A'
                    end_date = str(req['END_DATETIME']).split(' ')[0] if req['END_DATETIME'] else 'N/A'
                    reason_brief = req['REASON'][:20] + '...' if len(req['REASON']) > 20 else req['REASON']
                    
                    response += (
                        f"\n🔸 {i}번 (ID: {req['REQUEST_ID']}) | "
                        f"📅 {start_date}~{end_date} | "
                        f"📌 {req['TYPE_BIG']} | "
                        f"📝 {reason_brief}"
                    )
                
                response += f"\n\n{cancel_result['reason']}"
                response += "\n💡 취소하려면 'ID:숫자 취소'라고 말해주세요. 예: ID:123 취소"
                return response

        # ✅ 2단계: 신청 의도 처리 (조회보다 우선)
        if is_leave_intent(user_input):
            print("🧭 [휴가/공가/병가/조퇴 신청 의도 판단됨 → LLM 파싱 시도]")
            info = extract_leave_info(user_input)
            start = info.get("start_date")
            end = info.get("end_date")
            reason = info.get("reason")
            type_big = info.get("type_big") or "휴가"
            type_small = info.get("type_small") or "기타"

            if not (start and end and reason):
                return (
                    f"✨ {type_big}를 신청하시려는 것 같네요!\n\n"
                    "다음 정보를 함께 알려주세요:\n"
                    "📅 기간: 언제부터 언제까지인가요?\n"
                    "📝 사유: 어떤 이유인가요?\n\n"
                    "예시: '8월 1일부터 8월 3일까지 개인 사정으로 휴가 신청할래요'"
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
                    f"✅ {type_big} 신청이 정상적으로 접수되었습니다!\n\n"
                    f"📅 기간: {start} ~ {end}\n"
                    f"📌 유형: {type_big} / {type_small}\n"
                    f"📝 사유: {reason}\n\n"
                    f"승인까지 잠시 기다려주세요. 🙏"
                )
            else:
                return "❌ 신청 처리 중 오류가 발생했습니다. 다시 시도해주세요."

        # ✅ 3단계: 조회 의도 처리 (신청 의도 이후)
        if is_inquiry_intent(user_input):
            print("🔎 [조회 의도 감지됨]")
            
            # 3-1: 출결 신청 전체 내역 조회 (간결 버전)
            if "출결" in user_input:
                print("🔎 [출결 신청 전체 내역 조회 의도 감지됨]")
                all_records = get_attendance_records(student_id, limit=5)  # 상위 5개만
                return format_brief_attendance_records(all_records, "📋 최근 출결 신청 내역")
            
            # 3-2: 특정 유형별 신청 내역 조회
            for leave_type in ["휴가", "병가", "공가", "조퇴"]:
                if leave_type in user_input:
                    print(f"🔎 [{leave_type} 신청 내역 조회 의도 감지됨]")
                    type_records = get_attendance_records_by_type(student_id, leave_type)
                    
                    if not type_records:
                        return f"{leave_type} 신청 내역이 없습니다."
                    
                    return format_brief_attendance_records(type_records[:5], f"📋 {leave_type} 신청 내역")
            
            # 3-3: 일반 조회 (기존 호환성 유지)
            print("🔎 [일반 조회 의도 감지됨]")
            records = get_attendance_records(student_id, limit=5)  # 상위 5개만

            # 유형 필터 (호환성)
            filter_type = None
            for t in ["휴가", "병가", "공가", "조퇴"]:
                if t in user_input:
                    filter_type = t
                    break
            if filter_type:
                records = [r for r in records if r["TYPE_BIG"] == filter_type]

            if not records:
                return "최근 신청 내역이 없습니다."

            title = f"📋 최근 {filter_type} 신청 내역" if filter_type else "📋 최근 신청 내역"
            return format_brief_attendance_records(records, title)

        # ✅ 4단계: 일반 정보 질문 (RAG)
        print("🔍 [일반 정보 질의 → 문서 검색]")
        result = qa_chain(user_input)
        return str(result["result"])

    except Exception as e:
        print(f"[❌ 전체 처리 오류]: {e}")
        return "답변 중 오류가 발생했습니다. 다시 시도해주세요."