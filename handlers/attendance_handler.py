from langchain_openai import ChatOpenAI
from langchain.schema import Document  
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import Chroma  
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os 
from dotenv import load_dotenv

load_dotenv()

# API 키는 .env에서 자동 로드되므로 여기선 생략, temperature=0(결과 재현성 높게)로 생성
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.05)  

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

current_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(current_dir, ".."))

# 1) 파일 경로 리스트
paths = [
    os.path.join(base_dir, "utils", "trainee_admin_attendance_guide.txt"),
    os.path.join(base_dir, "utils", "training_handbook.txt")
]

# 2) 텍스트 스플리터 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,    # 한 덩어리 
    chunk_overlap=300   # 덩어리 간 겹치기  
)

# 3) Document 객체 생성 (분할 후)
docs: list[Document] = []
for p in paths:
    raw_text = open(p, encoding="utf-8").read()    # 원본 전체 텍스트 로드
    chunks = text_splitter.split_text(raw_text)    # 문단/문장 기준으로 쪼개기

    for idx, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source_file": p,   # 원본 파일 경로
                "chunk_index": idx  # 덩어리 번호
            }
        ))

# 3) ChromaDB에 임베딩 + 업로드
embeddings = OpenAIEmbeddings()  
vector_db = Chroma(persist_directory="./attendance_db", embedding_function=embeddings)
vector_db.add_documents(docs)  
vector_db.persist()  # 디스크에 저장


# 3) 검색 함수 정의
def retrieve_chunks(query: str, k: int = 5) -> list[Document]:

    # 출석 도메인의 다양한 용어들 매핑
    attendance_terms = {
        "QR": "출석체크 출결확인",
        "인정": "출석인정 출결인정"
    }
    # 검색에 실제로 사용할 확장된 질문 문자열을 저장하는 변수
    enhanced_query = query

    # term = key, expansion = value
    for term, expansion in attendance_terms.items():
        if term in query:
            enhanced_query += f" {expansion}"

    # vector_db는 미리 초기화된 ChromaDB 인스턴스
    # similarity_search로 query와 가장 유사한 k개의 문서 조각 반환
    return vector_db.similarity_search(enhanced_query, k=k)

# 4) 답변 생성 함수 정의
def generate_answer(query: str, chunks: list[Document]) -> str:
    docs_text = "\n\n".join(
        f"[{i+1}] (출처: {d.metadata['source_file']})\n{d.page_content}"
        for i, d in enumerate(chunks)
    )
    # system 메시지: 봇의 톤·룰(“출처 표시 방식”·“못 찾으면 사과”) 정의
    # user 메시지: 질문과 변환된 문서 조각 전달
    messages = [
        SystemMessage(content=(
            "당신은 패스트캠퍼스 수강생을 위한 행정 챗봇입니다. "
            "제공된 문서에서 근거를 찾아 명확히 답변하세요. "
            "정보가 없으면 문서에 근거해 안내하세요. 문서에 명시된 항목 외에는"
            "'대상이 아닙니다.'또는 '불가합니다.' 라고 명확히 답변하세요."
        )),
        HumanMessage(content=(
            f"사용자 질문: “{query}”\n\n"
            "관련 문서:\n"
            f"{docs_text}\n\n"
            "→ 답변:"
        ))
    ]
    return llm.predict_messages(messages).content.strip()

# 5) 전체 흐름 함수
def answer(query: str, top_k : int = 5) -> str:
    chunks = retrieve_chunks(query, k=top_k)      # ② 검색 단계
    return generate_answer(query, chunks)