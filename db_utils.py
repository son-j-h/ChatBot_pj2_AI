import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """MySQL 데이터베이스 연결 객체를 반환합니다."""
    print("DEBUG: 데이터베이스 연결 시도...")
    try:
        conn = mysql.connector.connect(
            host = os.getenv("MYSQL_HOST"),
            user = os.getenv("MYSQL_USER"),
            password = os.getenv("MYSQL_PASSWORD"),
            database = os.getenv("MYSQL_DB"),
            port=int(os.getenv("MYSQL_PORT", 4881))
        )
        print("DEBUG: 데이터베이스 연결 성공.")
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}") # 기존 오류 메시지 형식 유지
        return None

def get_student_info(student_id_str: str) -> dict or None:
    """
    주어진 학번(문자열)으로 학생 정보를 조회하여 딕셔너리 형태로 반환합니다.
    (예: {"STUDENT_NAME": "홍길동", "STUDENT_NUM": 2001})
    MySQL DDL에서 STUDENT_NUM이 INT 타입이므로, 변환하여 사용합니다.
    """
    print(f"DEBUG: get_student_info 함수 호출됨. 학번: '{student_id_str}'")
    conn = get_db_connection()
    if not conn:
        print("ERROR: 데이터베이스 연결 실패로 학생 정보를 조회할 수 없습니다.")
        return None

    try:
        # 문자열로 받은 학번을 INT 타입으로 변환
        try:
            student_id_int = int(student_id_str)
            print(f"DEBUG: 학번 '{student_id_str}'을(를) 정수형({student_id_int})으로 변환 성공.")
        except ValueError:
            print(f"ERROR: 학번 '{student_id_str}'을(를) 정수로 변환할 수 없습니다. 유효하지 않은 학번 형식입니다.")
            return None 

        cursor = conn.cursor(dictionary=True) # 딕셔너리 형태로 결과를 받기 위해 dictionary=True 설정
        # 쿼리를 수정하여 필요한 모든 학생 정보를 가져옵니다. (여기서는 이름과 학번)
        # 실제 DB 스키마에 맞게 필드명을 수정하세요.
        query = "SELECT STUDENT_NAME, STUDENT_NUM FROM STUDENT WHERE STUDENT_NUM = %s"
        print(f"DEBUG: 쿼리 실행: '{query}', 파라미터: {student_id_int}")
        cursor.execute(query, (student_id_int,)) # 변환된 INT 값을 쿼리에 전달
        student_info = cursor.fetchone() # 결과를 딕셔너리 형태로 받습니다.
        if student_info:
            # --- 핵심 로깅 시작 ---
            print(f"DEBUG: 학번 '{student_id_int}' 조회 성공. 학생 이름: {student_info.get('STUDENT_NAME')}")
            # --- 핵심 로깅 끝 ---
        else:
            # --- 핵심 로깅 시작 ---
            print(f"DEBUG: 학번 '{student_id_int}'에 해당하는 학생 정보를 찾을 수 없습니다.")
            # --- 핵심 로깅 끝 ---
        return student_info # 딕셔너리 또는 None 반환

    except mysql.connector.Error as err:
        print(f"Error fetching student info: {err}")
        return None

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()