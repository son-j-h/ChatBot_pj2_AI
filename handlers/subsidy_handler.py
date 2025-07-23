import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 벡터 DB 저장 위치 및 설정
PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../my_rag_db"))
COLLECTION_NAME = "admin_docs"

# ✅ 1. 벡터 DB 로딩만 수행 (생성 X)
def load_vectorstore():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

    if not os.path.exists(PERSIST_DIR):
        raise ValueError("❌ 벡터 DB 폴더가 존재하지 않습니다. 먼저 생성해 주세요.")

    print("📦 저장된 벡터 DB 로드 중...")

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})

# ✅ 2. 프롬프트 템플릿 정의
def get_subsidy_prompt():
    system_template = """너는 패스트캠퍼스의 훈련장려금 전문 상담 챗봇이야.
사용자의 질문에 대해 아래 참고 문서 내용만 기반으로 정확하고 친절하게 답변해.

- 참고 문서에 없는 정보는 "자료에 없음"이라고 말해.
- 핵심 정보를 간결하고 쉽게 설명해 줘.
- 필요한 경우 bullet list 형식으로 정리해 줘.
- 문서 내용을 직접 인용해도 좋아.

참고 문서:
{context}
"""
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{question}")
    ])

# ✅ 3. LCEL 체인 구성
def build_chain():
    retriever = load_vectorstore()

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=800,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = get_subsidy_prompt()

    chain = (
        {
            "context": lambda x: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["question"])]),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ✅ 4. answer() 함수
_chain = build_chain()

def answer(question: str) -> str:
    if not question.strip():
        return "질문을 입력해주세요."

    try:
        return _chain.invoke({"question": question})
    except Exception as e:
        print(f"[❌ 오류 발생]: {e}")
        return "답변 중 오류가 발생했습니다."