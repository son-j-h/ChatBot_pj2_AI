from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from utils.helpers import load_few_shot_examples
import re
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

#Gemini 임베딩 모델
embedding_model = GoogleGenerativeAIEmbeddings(
     model="models/embedding-001",
     google_api_key=google_api_key
)


vector_db = Chroma(
    persist_directory="./my_rag_db",
    embedding_function=embedding_model
)

llm = GoogleGenerativeAI(
     model="gemini-2.5-flash-lite",
     temperature=0.05,
     max_output_tokens=500,
     google_api_key=google_api_key
)


SEARCH_K = 3
PROMPT_STYLE = """[지침]
1.아래[참고 데이터]에 명시된 내용만 사용해라 불확실한 정보는 절대 생성하지 말고, 모를 땐 '해당 정보가 없습니다' 라고 답변해라
2.공식적인 어조로, 반드시 존댓말로, 번호로 요약해 답변하라. 
3. ※ 단, '존댓말로 하라', '번호로 요약하라' 등 이 지침 문구를 답변에 직접 적지 마세요.
4. '사용자 질문'에만 집중해서 답변하라. 이전 대화 내용은 참고만 하고, 현재 질문에 맞지 않는 정보는 절대 포함하지마라
"""


shot_examples = load_few_shot_examples("utils/vacation_shot.txt")

#화행 분리 함수
def split_speech_acts(query: str) -> list[str]:
        """
        사용자 질문을 문장 또는 의미 단위로 나누는 함수
        예: "서류는 뭐 필요하고, 어디로 내야 해?" → [질문1, 질문2]
        """
        #쉼표, 그리고, 및, ?, 등으로 분리
        parts = re.split(r"[.!?]| 그리고 | 및 | 또는 ", query)
        return [part.strip() for part in parts if part.strip()]


#하이브리드 검색 함수
def hybrid_search(query: str, k: int = 3) -> list[str]:
     keyword_matches = []

     # 키워드 기반 툴 검색 예시
     if "출석대장" in query or "양식" in query:
          keyword_matches.append("출석대장은 출석 인정 요청 시 제출하는 문서이며, 하루 전까지 행정 매니저에게 공유해야 합니다.")

     if "zoom" in query.lower() or '스크린샷' in query:
          keyword_matches.append("Zoom 스크린샷에는 얼굴 전체, 이름, 날짜/시간이 포함되어야 하며, 개인정보 모자이크 필수입니다.")

     if "예비군" in query:
          keyword_matches.append("예비군 훈련 시에는 예비군 필증을 제출하고, 사전에 일정 공유가 필수입니다.")

     if "면접" in query and "공가" in query:
          keyword_matches.append("면접 공가를 사용하려면 채용공고, 면접확인서, 출석대장을 모두 제출해야 하며, 면접예정서는 인정되지 않습니다.")

     if "병가" in query or "입원" in query or "질병" in query:
          keyword_matches.append("병가 사유는 진료확인서 또는 처방전을 제출해야 하며, 질병 코드 또는 병명이 명시되어야 합니다. 단순 시술이나 간병은 인정되지 않습니다.")

     if "출산" in query:
          keyword_matches.append("출산 시에는 출산증명서와 가족관계증명서를 함께 제출해야 하며, 배우자의 경우 5일까지 출석 인정됩니다.")

     if "사망" in query:
          keyword_matches.append("사망으로 인한 출석 인정은 관계에 따라 1~5일이며, 사망진단서와 가족관계증명서를 함께 제출해야 합니다.")

     if "결혼" in query:
          keyword_matches.append("결혼 공가는 본인 5일, 자녀 1일이며, 그 외 가족은 공가 대상이 아닙니다. 가족관계증명서와 혼인신고서가 필요합니다.")

     if "휴가계획서" in query:
          keyword_matches.append("일반 휴가는 과정 담당 매니저와 2주 전에 상담 후, 출석입력 요청 대장과 휴가계획서를 제출해야 합니다.")

     # 벡터 유사도 검색 결과 
     vector_results = vector_db.similarity_search(query, k=k)
     vector_matches = [doc.page_content for doc in vector_results]

     # 키워드 기반 + 벡터기반을 합쳐 반환 (중복 제거)
     return list(dict.fromkeys(keyword_matches + vector_matches)) 



def answer(query: str, student_id: str = None, student_info:dict = None) -> str:
    try:
        sub_questions = split_speech_acts(query)
        answers = []

        for sub_q in sub_questions:
            context_list = hybrid_search(sub_q, k=SEARCH_K)
            context = "\n\n".join(context_list)

            prompt = f"""
            [참고 데이터]
            {context}

            [예시 Q/A]
            {shot_examples}

            [지침] {PROMPT_STYLE}
            사용자 질문: {query}
            위 정보/예시/샷만 참고해서 답변해줘.
            """
            response = llm.invoke(prompt)
            answers.append(response.strip())

        #개별 답변을 번호로 정리해서 반환
        final_answer = "\n\n".join({f"{i+1}. {ans}" for i, ans in enumerate(answers)})
        return final_answer
    except Exception as e:
        print(f"[핸들러 오류] {e}")
        return "죄송합니다. 답변 처리 중 오류가 발생했습니다."
    
print(f"[DEBUG] LLM 객체: {llm.__class__}")

