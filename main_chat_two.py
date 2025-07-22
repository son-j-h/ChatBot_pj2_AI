from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from handlers import certificate_handler, leave_handler, vacation_handler, attendance_handler, subsidy_handler

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_key
)

tools = [
    Tool(
        name="CertificateHandler",
        func=certificate_handler.answer,
        description="수강증명서, 참가확인서, 출석부, 수료증 등 각종 증명서의 발급 절차 및 신청 방법에 대한 질문에 답변합니다. 예시: '수료증 발급 어떻게 받나요?', '참가확인서 신청 절차 알려주세요.', '출석부 증명서 발급 문의합니다.'"
    ),
    Tool(
        name="LeaveHandler",
        func=leave_handler.answer,
        description="사용자가 휴가, 조퇴, 병가를 *직접 신청하겠다고 요청할 때만* 답변합니다. 휴가/조퇴/병가의 절차나 규정에 대한 일반적인 문의는 VacationHandler에서 처리합니다. 예시: '다음 주 수요일에 병가 신청하고 싶어요', '오늘 오후에 조퇴 가능할까요?', '0월 0일에 휴가 신청해주세요.'"
    ),
    Tool(
        name="VacationHandler",
        func=vacation_handler.answer,
        description="개인 휴가, 예비군/병가/면접 등 기타 공가의 *종류, 사용 규정, 신청 절차, 관련 서류, 사용 일수 등 일반적인 정보나 절차에 대한 문의*에 답변합니다. 사용자가 휴가/조퇴/병가를 직접 신청하겠다고 요청하는 질문은 LeaveHandler에서 처리합니다. 예시: '병가 사용 규정이 어떻게 되나요?', '개인 휴가 일수는 얼마나 되나요?', '예비군 공가는 어떻게 신청하나요?', '공가 신청 시 어떤 서류가 필요한가요?'"
    ),
    Tool(
        name="AttendanceHandler",
        func=attendance_handler.answer,
        description="훈련생 출결 관리 안내, 출석체크 기본 규칙, 출결정정 신청 시 주의사항, HRD 오류 등으로 인한 QR 미체크 출결정정 신청 방법, 단순 외출 시 신청 방법 등 *출결 정정 및 관리에 대한 모든 질문*에 답변합니다. 예시: '출석체크는 어떻게 해야 하나요?', '출결 정정 신청 방법 알려주세요.', '외출할 때 신청해야 하나요?'"
    ),
    Tool(
        name="SubsidyHandler",
        func=subsidy_handler.answer,
        description="훈련장려금의 기본 지급 기준, 지급 금액, 장려금 지급 일정 및 *모든 훈련장려금 관련 규정*에 대한 질문에 답변합니다. 예시: '훈련장려금 얼마 받을 수 있나요?', '장려금 지급일이 언제인가요?', '훈련장려금 기본 지급 기준이 궁금해요.'"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # 디버깅용 로그 출력
)

app = Flask(__name__)
CORS(app)

@app.route("/answer", methods=["POST"])
def answer():
    print("answer() 함수 진입")
    data = request.get_json()
    user_input = data.get("message", "").strip()
    # 한국어로만 대답
    user_input = f"모든 답변은 한국어로 해주세요. 질문: {user_input}"

    if not user_input:
        return jsonify({"response": "질문을 입력해주세요."}), 400

    try:
        response_text = agent.run(user_input)
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"[❌ Agent 처리 오류]: {e}")
        return jsonify({"response": "답변 처리 중 오류가 발생했습니다."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
