from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json # JSON 파싱을 위해 추가
import datetime # 시간 정보 출력을 위해 추가

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate # 프롬프트 템플릿 사용을 위해 추가
from langchain.chains import LLMChain # LLM 체인 사용을 위해 추가

# 핸들러 모듈들을 임포트합니다.
# 실제 환경에서는 이 핸들러 파일들이 'handlers' 디렉토리 내에 존재해야 합니다.
from handlers import certificate_handler, leave_handler, vacation_handler, attendance_handler, subsidy_handler

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 1. 일반적인 정보성 질문에 답변할 LLM을 정의합니다. (기존 llm)
# 이 LLM은 각 핸들러 내부에서 사용되거나, 라우터/통합 LLM이 없을 경우의 폴백으로 사용될 수 있습니다.
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_key
)

# 2. 사용자 질문을 분해하고 의도를 분류할 '라우터 LLM'을 정의합니다.
# 이 LLM은 질문의 복잡성을 이해하고 여러 의도를 식별해야 하므로,
# 가능하다면 gpt-4o, gpt-4-turbo 등 더 강력한 모델을 사용하는 것을 권장합니다.
router_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", # 테스트 목적으로 gpt-3.5-turbo 사용, 실제 서비스 시 gpt-4 계열 권장
    temperature=0,
    openai_api_key=openai_key
)

# 3. 개별 답변들을 종합하여 최종 답변을 생성할 '통합 LLM'을 정의합니다.
synthesizer_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", # 테스트 목적으로 gpt-3.5-turbo 사용, 실제 서비스 시 gpt-4 계열 권장
    temperature=0.2, # 답변을 좀 더 유연하게 통합하기 위해 temperature를 약간 높일 수 있습니다.
    openai_api_key=openai_key
)

# 기존의 툴 정의는 그대로 유지합니다. 각 툴은 특정 도메인의 질문에 답변하는 역할을 합니다.
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

# Flask 애플리케이션을 초기화하고 CORS를 설정합니다.
app = Flask(__name__)
CORS(app)

# 로깅을 위한 헬퍼 함수
def log_progress(message: str):
    """현재 시간과 함께 진행 상황 메시지를 콘솔에 출력합니다."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

@app.route("/answer", methods=["POST"])
def answer():
    """
    사용자의 질문을 받아 다중 의도를 처리하고 통합된 답변을 반환하는 API 엔드포인트입니다.
    """
    log_progress("--- answer() 함수 진입 ---")
    data = request.get_json()
    user_input = data.get("message", "").strip()

    # 사용자 입력이 비어있는 경우 오류 응답을 반환합니다.
    if not user_input:
        log_progress("사용자 입력이 비어있습니다. 오류 응답 반환.")
        return jsonify({"response": "질문을 입력해주세요."}), 400

    log_progress(f"사용자 입력: '{user_input}'")

    try:
        # --------------------------------------------------------------------
        # 1단계: 질문 분해 및 의도 분류 (Intent Classification)
        # --------------------------------------------------------------------
        log_progress("1단계: 질문 분해 및 의도 분류 시작 (라우터 LLM 호출)")
        router_prompt_template = PromptTemplate(
            template="""
            당신은 사용자 질문을 분석하여 관련된 기능(tool)과 해당 기능에 전달할 질문을 분리하는 AI 비서입니다.
            아래에 정의된 tool들을 참고하여 사용자의 질문을 가장 적절하게 1개 이상의 tool과 sub_question으로 분리해주세요.
            만약 해당하는 tool이 없거나 질문의 의도를 명확히 알 수 없으면 tool_name을 'General'로 설정하고 sub_question에 원본 질문을 그대로 넣어주세요.
            결과는 반드시 JSON 형식의 배열로 반환해야 합니다.

            Tool Definitions:
            - CertificateHandler: {certificate_desc}
            - LeaveHandler: {leave_desc}
            - VacationHandler: {vacation_desc}
            - AttendanceHandler: {attendance_desc}
            - SubsidyHandler: {subsidy_desc}

            사용자 질문: {user_input}

            JSON 형식 예시:
            [
              {{"tool_name": "CertificateHandler", "sub_question": "수료증 발급 어떻게 받나요?"}},
              {{"tool_name": "VacationHandler", "sub_question": "병가 사용 규정이 어떻게 되나요?"}}
            ]
            """,
            input_variables=["user_input", "certificate_desc", "leave_desc", "vacation_desc", "attendance_desc", "subsidy_desc"]
        )

        router_chain = LLMChain(llm=router_llm, prompt=router_prompt_template)

        # 라우터 LLM을 호출하여 의도 분류 결과를 받습니다.
        raw_routing_output = router_chain.run(
            user_input=user_input,
            certificate_desc=tools[0].description,
            leave_desc=tools[1].description,
            vacation_desc=tools[2].description,
            attendance_desc=tools[3].description,
            subsidy_desc=tools[4].description
        )
        log_progress(f"라우터 LLM 원본 응답: {raw_routing_output}")

        parsed_intents = []
        try:
            # LLM 응답을 JSON으로 파싱합니다.
            parsed_intents = json.loads(raw_routing_output)
            # LLM이 때때로 단일 객체를 반환할 수 있으므로, 리스트가 아니면 리스트로 감쌉니다.
            if not isinstance(parsed_intents, list):
                parsed_intents = [parsed_intents]
            log_progress(f"파싱된 의도: {parsed_intents}")
        except json.JSONDecodeError:
            log_progress(f"❌ 라우터 LLM 출력 JSON 파싱 실패: {raw_routing_output}. 전체 질문을 'General' 의도로 처리합니다.")
            # JSON 파싱 실패 시, 전체 질문을 'General' 의도로 처리하는 폴백 로직
            parsed_intents = [{"tool_name": "General", "sub_question": user_input}]
        
        # 파싱된 의도가 없으면 (예: 빈 리스트 반환) 'General' 의도로 처리합니다.
        if not parsed_intents:
            log_progress("파싱된 의도가 없습니다. 전체 질문을 'General' 의도로 처리합니다.")
            parsed_intents = [{"tool_name": "General", "sub_question": user_input}]

        individual_responses = [] # 각 핸들러에서 받은 답변들을 저장할 리스트

        # --------------------------------------------------------------------
        # 2단계: 개별 핸들러 실행 (Handler Execution)
        # --------------------------------------------------------------------
        log_progress("2단계: 개별 핸들러 실행 시작")
        for i, intent_info in enumerate(parsed_intents):
            tool_name = intent_info.get("tool_name")
            sub_question = intent_info.get("sub_question")
            log_progress(f"  [{i+1}/{len(parsed_intents)}] 처리 중 의도: tool_name='{tool_name}', sub_question='{sub_question}'")

            # tool_name 또는 sub_question이 유효하지 않으면 건너뜁니다.
            if not tool_name or not sub_question:
                log_progress(f"  경고: 유효하지 않은 의도 정보 스킵: {intent_info}")
                continue

            if tool_name == "General":
                # 'General' 의도는 특정 핸들러에 매핑되지 않는 질문입니다.
                general_response = f"'{sub_question}'에 대한 특정 정보를 찾기 어렵습니다. 다른 질문이 있으신가요?"
                individual_responses.append(general_response)
                log_progress(f"  'General' 의도 처리 완료. 응답: '{general_response}'")
                continue

            # tools 리스트에서 해당하는 Tool 객체를 찾습니다.
            target_tool = next((t for t in tools if t.name == tool_name), None)

            if target_tool:
                log_progress(f"  '{tool_name}' 핸들러 호출 중...")
                try:
                    # 해당 핸들러의 'func' (answer 함수)를 직접 호출합니다.
                    tool_response = target_tool.func(sub_question)
                    individual_responses.append(tool_response)
                    log_progress(f"  '{tool_name}' 핸들러 응답: '{tool_response}'")
                except Exception as tool_e:
                    log_progress(f"  [❌ {tool_name} 핸들러 오류]: {tool_e}")
                    individual_responses.append(f"'{sub_question}' 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            else:
                # 라우터 LLM이 존재하지 않는 툴 이름을 반환한 경우
                error_msg = f"'{sub_question}' 질문에 해당하는 처리기({tool_name})를 찾을 수 없습니다."
                individual_responses.append(error_msg)
                log_progress(f"  ❌ {error_msg}")
        
        log_progress(f"모든 개별 핸들러 실행 완료. 수집된 개별 답변: {individual_responses}")

        # --------------------------------------------------------------------
        # 3단계: 답변 통합 (Response Synthesis)
        # --------------------------------------------------------------------
        log_progress("3단계: 답변 통합 시작 (통합 LLM 호출)")
        if not individual_responses:
            # 처리된 답변이 하나도 없는 경우
            final_response = "죄송합니다. 질문을 이해하지 못했습니다. 더 자세히 알려주시겠어요?"
            log_progress("처리된 답변이 없어 기본 폴백 응답을 사용합니다.")
        else:
            # 통합 LLM을 사용하여 개별 답변들을 자연스럽게 연결하고 통합합니다.
            synthesis_prompt_template = PromptTemplate(
                template="""
                다음은 사용자의 여러 질문에 대한 개별적인 답변들입니다. 이 답변들을 조합하여 하나의 자연스럽고 유려한 한국어 문장으로 만들어주세요.
                각 답변의 내용이 명확하게 드러나도록 하고, 필요한 경우 연결어를 사용하여 부드럽게 이어주세요.
                존댓말을 사용해주세요.

                개별 답변들:
                {individual_responses_str}

                최종 답변:
                """,
                input_variables=["individual_responses_str"]
            )
            synthesizer_chain = LLMChain(llm=synthesizer_llm, prompt=synthesis_prompt_template)

            # 개별 답변 리스트를 문자열로 변환하여 프롬프트에 전달합니다.
            individual_responses_str = "\n- ".join(individual_responses)
            if individual_responses_str:
                individual_responses_str = "- " + individual_responses_str
            
            log_progress(f"통합 LLM에 전달할 개별 답변 문자열: \n{individual_responses_str}")
            final_response = synthesizer_chain.run(individual_responses_str=individual_responses_str)
            log_progress(f"통합 LLM 최종 응답: {final_response}")

        # 최종 답변을 한국어로만 제공하도록 보장합니다. (통합 LLM이 이미 한국어로 생성하므로 중복될 수 있습니다.)
        # final_response = f"모든 답변은 한국어로 제공됩니다. {final_response.strip()}"

        log_progress("--- answer() 함수 종료 ---")
        return jsonify({"response": final_response.strip()}) # 불필요한 공백 제거

    except Exception as e:
        # 전체 처리 과정에서 예상치 못한 오류가 발생한 경우
        log_progress(f"[❌ 전체 처리 오류]: {e}")
        return jsonify({"response": "답변 처리 중 예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}), 500

if __name__ == "__main__":
    # Flask 애플리케이션을 실행합니다.
    app.run(host="0.0.0.0", port=5001, debug=True)