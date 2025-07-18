# main_chat.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from handlers import certificate_handler, leave_handler  # attendance_handler 필요시 추가
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# LLM 인스턴스
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    openai_api_key=openai_key
)

app = Flask(__name__)
CORS(app)


# 🔁 백업용 키워드 기반 주제 분류
def classify_topic_fallback(user_input: str) -> str:
    if any(keyword in user_input for keyword in ["수강증명서", "참가확인서", "출석부", "발급"]):
        return "certificate"
    # elif any(keyword in user_input for keyword in ["출결정정", "지각"]):
    #     return "attendance"
    elif any(keyword in user_input for keyword in ["휴가", "공가", "병가", "조퇴"]):
        return "leave"
    return "default"


# 🧠 LLM 기반 + fallback 분류
def classify_topic(user_input: str) -> str:
    prompt = f"""
다음 문장의 주제를 하나로 분류해줘. 가능한 주제는 다음 중 하나야:

- certificate (증명서/출석 관련)
- attendance (출결 정정/지각 관련)
- leave (휴가/조퇴/병가 관련)
- default (일반 질문)

문장: "{user_input}"

위 문장은 어떤 주제인지 딱 하나만 골라서 아래 형식처럼 알려줘:
답: <주제>
"""
    try:
        response = llm.predict(prompt).strip().lower()
        print(f"[🔍 주제 분류 결과]: {response}")
        for topic in ["certificate", "attendance", "leave", "default"]:
            if topic in response:
                return topic
        return "default"
    except Exception as e:
        print(f"[❌ 분류 오류 → fallback 사용]: {e}")
        return classify_topic_fallback(user_input)


@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "질문을 입력해주세요."}), 400

    topic = classify_topic(user_input)

    try:
        if topic == "certificate":
            response_text = certificate_handler.answer(user_input)
        elif topic == "leave":
            response_text = leave_handler.answer(user_input)
        # elif topic == "attendance":
        #     response_text = attendance_handler.answer(user_input)
        else:
            response_text = "🤖 이 질문은 아직 지원하지 않아요. 다시 질문해 주세요."

        return jsonify({"response": response_text})

    except Exception as e:
        print(f"[❌ 핸들러 처리 오류]: {e}")
        return jsonify({"response": "답변 처리 중 오류가 발생했습니다."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
