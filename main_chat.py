from handlers import certificate_handler, attendance_handler, leave_handler
from dotenv import load_dotenv
import os

load_dotenv()

def classify_topic(user_input: str) -> str:
    """입력된 질문을 기반으로 담당 파트를 분류"""
    if "증명서" in user_input:
        return "certificate"
    elif "출결정정" in user_input or "지각" in user_input:
        return "attendance"
    elif "휴가" in user_input or "조퇴" in user_input or "병가" in user_input:
        return "leave"
    # 나머지도 추가
    return "default"

if __name__ == "__main__":
    print("🎓 패캠 행정 챗봇입니다. 무엇이든 물어보세요! (종료하려면 '그만')")
    while True:
        query = input("\n👤 사용자: ")
        if query.lower() in ["그만", "exit", "quit"]:
            print("👋 챗봇을 종료합니다.")
            break

        topic = classify_topic(query)

        if topic == "certificate":
            certificate_handler.answer(query)
        elif topic == "attendance":
            attendance_handler.answer(query)
        elif topic == "leave":
            leave_handler.answer(query)
        else:
            print("🤖 이 질문은 아직 지원하지 않아요. 다시 질문해 주세요.")