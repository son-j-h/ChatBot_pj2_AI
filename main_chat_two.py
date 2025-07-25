from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.helpers import load_text
from dotenv import load_dotenv
import os
import json  # JSON 파싱을 위해 추가
import datetime  # 시간 정보 출력을 위해 추가
import re
import logging


from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate  # 프롬프트 템플릿 사용을 위해 추가
from langchain.chains import LLMChain  # LLM 체인 사용을 위해 추가
from langchain_core.prompts import PromptTemplate

# 핸들러 모듈들을 임포트합니다.
# 실제 환경에서는 이 핸들러 파일들이 'handlers' 디렉토리 내에 존재해야 합니다.
from handlers import (
    certificate_handler,
    leave_handler,
    vacation_handler,
    attendance_handler,
    subsidy_handler,
)
from db_utils import get_student_info


# ✨ 실시간 벡터 메모리 활용을 위한 import
from utils.chat_history import save_chat_to_vectorstore, retrieve_context

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 1. 일반적인 정보성 질문에 답변할 LLM을 정의합니다. (기존 llm)
# 이 LLM은 각 핸들러 내부에서 사용되거나, 라우터/통합 LLM이 없을 경우의 폴백으로 사용될 수 있습니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_key)

# 2. 사용자 질문을 분해하고 의도를 분류할 '라우터 LLM'을 정의합니다.
# 이 LLM은 질문의 복잡성을 이해하고 여러 의도를 식별해야 하므로,
# 가능하다면 gpt-4o, gpt-4-turbo 등 더 강력한 모델을 사용하는 것을 권장합니다.
router_llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_key)

# 3. 개별 답변들을 종합하여 최종 답변을 생성할 '통합 LLM'을 정의합니다.
synthesizer_llm = ChatOpenAI(
    model_name="gpt-4o", temperature=0.2, openai_api_key=openai_key
)

# 기존의 툴 정의는 그대로 유지합니다. 각 툴은 특정 도메인의 질문에 답변하는 역할을 합니다.
tools = [
    Tool(
        name="CertificateHandler",
        func=certificate_handler.answer,
        description="수강증명서, 참가확인서, 출석부, 수료증 등 각종 증명서의 발급 절차 및 신청 방법에 대한 질문에 답변합니다. 예시: '수료증 발급 어떻게 받나요?', '참가확인서 신청 절차 알려주세요.', '출석부 증명서 발급 문의합니다.'",
    ),
    Tool(
        name="LeaveHandler",
        func=leave_handler.answer,
        description="사용자가 휴가, 조퇴, 병가를 *직접 신청하겠다고 요청할 때만* 답변합니다. 휴가/조퇴/병가의 절차나 규정에 대한 일반적인 문의는 VacationHandler에서 처리합니다. 예시: '다음 주 수요일에 병가 신청하고 싶어요', '오늘 오후에 조퇴 가능할까요?', '0월 0일에 휴가 신청해주세요.'",
    ),
    Tool(
        name="VacationHandler",
        func=vacation_handler.answer,
        description="개인 휴가, 예비군/병가/면접 등 기타 공가의 *종류, 사용 규정, 신청 절차, 관련 서류, 사용 일수 등 일반적인 정보나 절차에 대한 문의*에 답변합니다. 사용자가 휴가/조퇴/병가를 직접 신청하겠다고 요청하는 질문은 LeaveHandler에서 처리합니다. 예시: '병가 사용 규정이 어떻게 되나요?', '개인 휴가 일수는 얼마나 되나요?', '예비군 공가는 어떻게 신청하나요?', '공가 신청 시 어떤 서류가 필요한가요?'",
    ),
    Tool(
        name="AttendanceHandler",
        func=attendance_handler.answer,
        description="훈련생 출결 관리 안내, 출석체크 기본 규칙, 출결정정 신청 시 주의사항, HRD 오류 등으로 인한 QR 미체크 출결정정 신청 방법, 단순 외출 시 신청 방법 등 *출결 정정 및 관리에 대한 모든 질문*에 답변합니다. 예시: '출석체크는 어떻게 해야 하나요?', '출결 정정 신청 방법 알려주세요.', '외출할 때 신청해야 하나요?'",
    ),
    Tool(
        name="SubsidyHandler",
        func=subsidy_handler.answer,
        description="훈련장려금의 기본 지급 기준, 지급 금액, 장려금 지급 일정 및 *모든 훈련장려금 관련 규정*에 대한 질문에 답변합니다. 예시: '훈련장려금 얼마 받을 수 있나요?', '장려금 지급일이 언제인가요?', '훈련장려금 기본 지급 기준이 궁금해요.'",
    ),
]

# Flask 애플리케이션을 초기화하고 CORS를 설정합니다.
app = Flask(__name__)
CORS(app)


# 로깅을 위한 헬퍼 함수
def log_progress(message: str):
    """현재 시간과 함께 진행 상황 메시지를 콘솔에 출력합니다."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


# 각 세션 ID별로 학번, 학생 정보, 대화 상태를 저장
session_data = {}
STATE_INITIAL = "initial"  # 챗봇 시작 및 학번 요청
STATE_ID_PENDING = "id_pending"  # 학번 입력 대기 중
STATE_CONVERSATION_ACTIVE = "conversation_active"  # 일반적인 대화 중


def extract_student_id(user_input: str) -> str or None:
    """
    사용자 입력에서 학번 패턴을 추출합니다.
    숫자만으로 이루어진 4자리 문자열을 학번으로 간주하는 정규 표현식.
    """
    match = re.search(r"(?:학번은?|학번)\s*(\d{4})", user_input)
    if match:
        print(
            f"DEBUG: 학번 추출 성공 (키워드 기반): '{match.group(1)}' from '{user_input}'"
        )
        return match.group(1)

    # 키워드 없이 단순히 숫자만 입력했을 경우도 고려하여 fallback
    # 단어 경계가 없는 숫자 패턴을 찾습니다.
    match_fallback = re.search(r"(\d{4})", user_input)
    if match_fallback:
        print(
            f"DEBUG: 학번 추출 성공 (숫자만): '{match_fallback.group(1)}' from '{user_input}'"
        )
        return match_fallback.group(1)

    print(f"DEBUG: 학번 추출 실패: '{user_input}'에서 학번 패턴을 찾을 수 없음.")
    return None


@app.route("/answer", methods=["POST"])
def answer():
    intermediate_messages = []
    log_progress("--- answer() 함수 진입 ---")
    data = request.get_json()
    user_input = data.get("message", "").strip()
    # 클라이언트에서 세션 ID를 'X-Session-ID' 헤더로 보내도록 가정
    # 클라이언트(프론트엔드)에서는 고유한 세션 ID를 생성하여 요청마다 포함해야 합니다.
    session_id = request.headers.get("X-Session-ID", "default_session")

    # 세션 데이터 초기화 또는 로드
    current_session = session_data.setdefault(
        session_id, {"state": STATE_INITIAL, "student_id": None, "student_info": None}
    )

    # 사용자 입력이 비어있는 경우 오류 응답을 반환합니다.
    if not user_input:
        if current_session["state"] == STATE_INITIAL:
            # 첫 진입 시 메시지 반환 (클라이언트에서 빈 메시지로 호출한다고 가정)
            current_session["state"] = STATE_ID_PENDING
            log_progress(f"세션 {session_id}: 초기 상태. 학번 요청 메시지 반환.")
            return jsonify(
                {
                    "response": "안녕하세요. 패캠 행정문의 챗봇 '우주🌌🧑‍🚀' 입니다. 학번을 말해주세요."
                }
            )
        else:
            log_progress(
                f"세션 {session_id}: 사용자 입력이 비어있습니다. 오류 응답 반환."
            )
            return jsonify({"response": "질문을 입력해주세요."}), 400

    log_progress(
        f"세션 {session_id} - 사용자 입력: '{user_input}', 현재 상태: {current_session['state']}"
    )

    # --------------------------------------------------------------------
    # 학번 입력 대기 상태 처리
    # --------------------------------------------------------------------
    if (
        current_session["state"] == STATE_ID_PENDING
        or current_session["student_id"] is None
    ):
        log_progress(f"세션 {session_id}: 학번 입력 대기 상태 또는 학번 미확인 상태.")
        extracted_id = extract_student_id(user_input)

        if extracted_id:
            log_progress(f"세션 {session_id}: 입력에서 학번 '{extracted_id}' 추출됨.")
            student_info = get_student_info(extracted_id)

            if student_info:
                student_name = student_info.get("STUDENT_NAME", "훈련생")
                current_session["student_id"] = extracted_id
                current_session["student_info"] = student_info
                current_session["state"] = STATE_CONVERSATION_ACTIVE
                log_progress(
                    f"세션 {session_id}: 학번 '{extracted_id}' ({student_name}) 확인. 대화 상태로 전환."
                )
                return jsonify(
                    {"response": f"{student_name}님, 어떤 것이 궁금하신가요?"}
                )
            else:
                log_progress(
                    f"세션 {session_id}: 학번 '{extracted_id}'로 학생 정보를 찾을 수 없음."
                )
                return jsonify(
                    {
                        "response": f"입력하신 학번({extracted_id})으로 학생 정보를 찾을 수 없습니다. 정확한 학번을 다시 알려주시겠어요?"
                    }
                )
        else:
            # 학번 입력 대기 중인데, 학번이 아닌 다른 질문을 했을 경우
            log_progress(
                f"세션 {session_id}: 학번 대기 중인데, 학번 패턴을 찾을 수 없음."
            )
            return jsonify(
                {
                    "response": "죄송합니다. 먼저 학번을 알려주세요. 학번은 4자리의 숫자로 입력해주세요.",
                }
            )

    # --------------------------------------------------------------------
    # 학번이 확인된 후 일반적인 대화 처리 (기존 answer 함수의 핵심 로직)
    # --------------------------------------------------------------------
    try:
        current_student_id = current_session["student_id"]
        student_info = current_session["student_info"]


        # ✅ RAG 문맥 검색: 과거 대화 히스토리 불러오기
        rag_context_docs = retrieve_context(user_input, student_id=current_student_id)
        rag_context = "\n".join([doc.page_content for doc in rag_context_docs])

        # ✅ RAG 문맥 검색: 과거 대화 히스토리 불러오기
        rag_context_docs = retrieve_context(user_input, student_id=current_student_id)
        rag_context = "\n".join([doc.page_content for doc in rag_context_docs])

        # 1단계: 질문 분해 및 의도 분류 (Intent Classification)
        log_progress("1단계: 질문 분해 및 의도 분류 시작 (라우터 LLM 호출)")
        intermediate_messages.append("질문 내용을 분석하고 있어요...")
        router_prompt_template = PromptTemplate(
            template="""
            당신은 사용자 질문을 분석하여 관련된 기능(tool)과 해당 기능에 전달할 질문을 분리하는 AI 비서입니다.
            아래에 정의된 tool들을 참고하여 사용자의 질문을 가장 적절하게 1개 이상의 tool과 sub_question으로 분리해주세요.
            **만약 사용자의 질문에 특정 개인 정보를 요구하는 질문(예: '내 수료증 발급', '내 훈련 장려금', '내 휴가')이라면,
            tool_name을 'RequireStudentID'로 설정하고 sub_question에 '학번이 필요합니다.'라고 명시해주세요.**
            (현재 사용자의 학번이 {current_student_id}로 확인되었으므로, 'RequireStudentID'로 분류된 질문도 이 학번을 사용하여 처리할 수 있습니다.)
            만약 해당하는 tool이 없거나 질문의 의도를 명확히 알 수 없으면 tool_name을 'General'로 설정하고 sub_question에 원본 질문을 그대로 넣어주세요.
            결과는 반드시 JSON 형식의 배열로 반환해야 합니다.

            Tool Definitions:
            - CertificateHandler: {certificate_desc}
            - LeaveHandler: {leave_desc}
            - VacationHandler: {vacation_desc}
            - AttendanceHandler: {attendance_desc}
            - SubsidyHandler: {subsidy_desc}
            - RequireStudentID: 특정 개인 정보 조회를 위해 학번이 필요한 경우. 예시: '내 훈련 장려금 알려줘', '내 휴가 신청해줘', '나의 수강증명서 발급받고 싶어.'

            사용자 질문: {user_input}

            JSON 형식 예시:
            [
              {{"tool_name": "CertificateHandler", "sub_question": "수료증 발급 어떻게 받나요?"}},
              {{"tool_name": "RequireStudentID", "sub_question": "학번이 필요합니다."}},
              {{"tool_name": "VacationHandler", "sub_question": "병가 사용 규정이 어떻게 되나요?"}}
            ]
            """,
            input_variables=[
                "user_input",
                "certificate_desc",
                "leave_desc",
                "vacation_desc",
                "attendance_desc",
                "subsidy_desc",
            ],
        )

        router_chain = LLMChain(llm=router_llm, prompt=router_prompt_template)

        # 라우터 LLM을 호출하여 의도 분류 결과를 받습니다.
        raw_routing_output = router_chain.run(
            user_input=user_input,
            certificate_desc=tools[0].description,
            leave_desc=tools[1].description,
            vacation_desc=tools[2].description,
            attendance_desc=tools[3].description,
            subsidy_desc=tools[4].description,
            current_student_id=current_student_id,  # 현재 학번을 프롬프트에 전달
        )
        log_progress(f"라우터 LLM 원본 응답: {raw_routing_output}")

        # --- 이 부분이 중요합니다. JSON 파싱 전에 전처리! ---
        processed_router_llm_output = raw_routing_output.strip()

        # 만약 응답이 ```json 으로 시작하고 ``` 로 끝난다면 제거
        if processed_router_llm_output.startswith(
            "```json"
        ) and processed_router_llm_output.endswith("```"):
            # ```json 와 ``` 를 제거하고, 내부의 줄바꿈과 공백을 정리 (strip)
            processed_router_llm_output = processed_router_llm_output[
                len("```json") : -len("```")
            ].strip()
            log_progress(
                f"DEBUG: 백틱 제거 후 전처리된 LLM 응답: {processed_router_llm_output}"
            )
        # --- 수정 끝 ---

        parsed_intents = []
        try:
            # LLM 응답을 JSON으로 파싱합니다.
            parsed_intents = json.loads(processed_router_llm_output)
            # LLM이 때때로 단일 객체를 반환할 수 있으므로, 리스트가 아니면 리스트로 감쌉니다.
            if not isinstance(parsed_intents, list):
                parsed_intents = [parsed_intents]
            log_progress(f"파싱된 의도: {parsed_intents}")
            
            # 분류된 카테고리 이름을 사용자에게 표시
            tool_names = [intent['tool_name'] for intent in parsed_intents if intent['tool_name'] != 'General']
            if tool_names:
                display_tools = ", ".join(tool_names)
                intermediate_messages.append(f"문의하신 내용을 '{display_tools}' 관련으로 분류했습니다.")
            else:
                intermediate_messages.append("질문 의도를 파악했습니다.")
                
        except json.JSONDecodeError:
            log_progress(
                f"❌ 라우터 LLM 출력 JSON 파싱 실패: {processed_router_llm_output}. 전체 질문을 'General' 의도로 처리합니다."
            )
            # JSON 파싱 실패 시, 전체 질문을 'General' 의도로 처리하는 폴백 로직
            parsed_intents = [{"tool_name": "General", "sub_question": user_input}]
            intermediate_messages.append("질문 분석에 문제가 발생하여 일반적인 방법으로 답변을 준비합니다.")

        # 파싱된 의도가 없으면 (예: 빈 리스트 반환) 'General' 의도로 처리합니다.
        if not parsed_intents:
            log_progress(
                "파싱된 의도가 없습니다. 전체 질문을 'General' 의도로 처리합니다."
            )
            parsed_intents = [{"tool_name": "General", "sub_question": user_input}]
            intermediate_messages.append("질문 의도를 파악하지 못했습니다. 일반적인 답변을 준비합니다.")

        individual_responses = []  # 각 핸들러에서 받은 답변들을 저장할 리스트

        # --------------------------------------------------------------------
        # 2단계: 개별 핸들러 실행 (Handler Execution)
        # --------------------------------------------------------------------
        log_progress("2단계: 개별 핸들러 실행 시작")
        for i, intent_info in enumerate(parsed_intents):
            tool_name = intent_info.get("tool_name")
            sub_question = intent_info.get("sub_question")
            log_progress(
                f"  [{i+1}/{len(parsed_intents)}] 처리 중 의도: tool_name='{tool_name}', sub_question='{sub_question}'"
            )
            intermediate_messages.append(f"'{tool_name}' 관련 정보를 조회하고 있어요...") # 개별 조회 시작 메시지


            # tool_name 또는 sub_question이 유효하지 않으면 건너뜁니다.
            if not tool_name or not sub_question:
                log_progress(f"  경고: 유효하지 않은 의도 정보 스킵: {intent_info}")
                continue

            # 'RequireStudentID'는 주로 챗봇이 사용자의 학번을 모르는 초기 단계에서 학번 입력을 유도하기 위한 의도입니다.
            # 하지만 현재는 사용자의 학번(current_student_id)이 이미 세션에 성공적으로 확인된 상태입니다.
            # 따라서 이 경우 학번을 다시 요청하는 것은 불필요하며, 대화 흐름을 방해할 수 있습니다.
            # 대신, 학번이 이미 확인되었음을 사용자에게 알리고, 해당 'sub_question'에 대한 답변 처리를
            # 계속 진행할 것임을 나타내는 메시지를 'individual_responses'에 추가합니다.
            # 이 메시지는 최종 답변 통합 단계에서 다른 답변들과 함께 자연스럽게 연결될 것입니다.
            if tool_name == "RequireStudentID":
                individual_responses.append(
                    f"학번({current_student_id})은 이미 확인되었습니다. '{sub_question}' 질문에 대한 답변을 준비합니다."
                )
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
                    tool_args = {
                        "question": sub_question,
                        "student_id": current_student_id,
                        "student_info": student_info,
                    }
                    tool_response = target_tool.func(sub_question)
                    individual_responses.append(tool_response)
                    log_progress(f"  '{tool_name}' 핸들러 응답: '{tool_response}'")
                except Exception as tool_e:
                    log_progress(f"  [❌ {tool_name} 핸들러 오류]: {tool_e}")
                    individual_responses.append(
                        f"'{sub_question}' 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                    )
            else:
                # 라우터 LLM이 존재하지 않는 툴 이름을 반환한 경우
                error_msg = f"'{sub_question}' 질문에 해당하는 처리기({tool_name})를 찾을 수 없습니다."
                individual_responses.append(error_msg)
                log_progress(f"  ❌ {error_msg}")

        log_progress(
            f"모든 개별 핸들러 실행 완료. 수집된 개별 답변: {individual_responses}"
        )
        intermediate_messages.append("수집된 정보를 통합하여 답변을 정리하고 있어요...") # 답변 통합 시작 메시지

        # --------------------------------------------------------------------
        # 3단계: 답변 통합 (Response Synthesis)
        # --------------------------------------------------------------------
        log_progress("3단계: 답변 통합 시작 (통합 LLM 호출)")
        if not individual_responses:
            # 처리된 답변이 하나도 없는 경우
            final_response = (
                "죄송합니다. 질문을 이해하지 못했습니다. 더 자세히 알려주시겠어요?"
            )
            log_progress("처리된 답변이 없어 기본 폴백 응답을 사용합니다.")
        else:
            # 🔄 기존 통합 프롬프트를 RAG context 포함 버전으로 교체
            synthesis_prompt_template = PromptTemplate(
                template="""
                다음은 사용자의 여러 질문에 대한 개별적인 답변들입니다.
                사용자의 최근 대화 문맥도 아래에 포함되어 있습니다.
                대화 문맥을 참고하여 답변들이 자연스럽게 이어지도록 유려하게 하나의 한국어 문장으로 만들어주세요.
                존댓말을 사용해주세요.

                최근 대화 문맥:
                {rag_context}

                개별 답변들:
                {individual_responses_str}

                최종 답변:
                """,
                input_variables=["rag_context", "individual_responses_str"],
            )
            synthesizer_chain = LLMChain(
                llm=synthesizer_llm, prompt=synthesis_prompt_template
            )

            # 개별 답변 리스트를 문자열로 변환하여 프롬프트에 전달합니다.
            individual_responses_str = "\n- ".join(individual_responses)
            if individual_responses_str:
                individual_responses_str = "- " + individual_responses_str

            log_progress(
                f"통합 LLM에 전달할 개별 답변 문자열: \n{individual_responses_str}"
            )
            final_response = synthesizer_chain.run(
                rag_context=rag_context,
                individual_responses_str=individual_responses_str,
            )
            log_progress(f"통합 LLM 최종 응답: {final_response}")

        # ✨ 실시간 대화 저장
        save_chat_to_vectorstore(
            user_input, final_response, student_id=current_student_id
        )
        log_progress("실시간 대화 저장 완료.")

        # 최종 답변을 한국어로만 제공하도록 보장합니다. (통합 LLM이 이미 한국어로 생성하므로 중복될 수 있습니다.)
        # final_response = f"모든 답변은 한국어로 제공됩니다. {final_response.strip()}"

        log_progress("--- answer() 함수 종료 ---")
        return jsonify({"response": final_response.strip(), "intermediate_messages": intermediate_messages})  # 불필요한 공백 제거

    except Exception as e:
        # 전체 처리 과정에서 예상치 못한 오류가 발생한 경우
        log_progress(f"[❌ 전체 처리 오류]: {e}")
        return (
            jsonify(
                {
                    "response": "답변 처리 중 예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                    "intermediate_messages": "요청 처리 중 오류가 발생했습니다."
                }
            ),
            500,
        )


if __name__ == "__main__":
    # Flask 애플리케이션을 실행합니다.
    app.run(host="0.0.0.0", port=5001, debug=True)