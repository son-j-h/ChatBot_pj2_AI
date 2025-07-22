from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from handlers import certificate_handler, leave_handler  # attendance_handler 추가 가능

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
        description="수강증명서, 참가확인서, 출석부, 증명서 발급에 대한 질문에 대답합니다."
    ),
    Tool(
        name="LeaveHandler",
        func=leave_handler.answer,
        description="휴가, 조퇴, 병가, 공가 관련 질문에 대답합니다."
    ),
    # 추가해보겠습니다~~~
    Tool(
        name="AttendanceHandler",
        func=attendance_handler.answer,
        description="출결정정 관련 질문에 대답합니다."
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
