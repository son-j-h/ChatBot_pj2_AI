from flask import Flask, request, jsonify
from handlers import certificate_handler, attendance_handler, leave_handler
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


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
        return jsonify({"response": certificate_handler.answer(user_input)})
    elif topic == "attendance":
        return jsonify({"response": attendance_handler.answer(user_input)})
    elif topic == "leave":
        return jsonify({"response": leave_handler.answer(user_input)})
    else:
        return jsonify(
            {"response": "🤖 이 질문은 아직 지원하지 않아요. 다시 질문해 주세요."}
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
