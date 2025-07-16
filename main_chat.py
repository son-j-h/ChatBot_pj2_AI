from flask import Flask, request, jsonify
from flask_cors import CORS
from handlers import (
    certificate_handler,
)  # attendance_handler, leave_handler도 같은 방식으로 준비

app = Flask(__name__)
CORS(app)  # 모든 도메인에서 오는 요청 허용 (개발환경용)


def classify_topic(user_input: str) -> str:
    if (
        "수강증명서" in user_input
        or "참가확인서" in user_input
        or "출석부" in user_input
        or "발급" in user_input
    ):
        return "certificate"
    elif "출결정정" in user_input or "지각" in user_input:
        return "attendance"
    elif "휴가" in user_input or "조퇴" in user_input or "병가" in user_input:
        return "leave"
    return "default"


@app.route("/answer", methods=["POST"])
def answer():
    user_input = request.json.get("message", "")
    topic = classify_topic(user_input)

    if topic == "certificate":
        response_text = certificate_handler.answer(user_input)
    # elif topic == "attendance":
    #     response_text = attendance_handler.answer(user_input)
    # elif topic == "leave":
    #     response_text = leave_handler.answer(user_input)
    else:
        response_text = "🤖 이 질문은 아직 지원하지 않아요. 다시 질문해 주세요."

    return jsonify({"response": response_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
