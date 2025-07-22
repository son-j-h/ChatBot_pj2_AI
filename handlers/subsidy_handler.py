import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 0. 환경변수 로드
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 벡터 DB 저장 위치
PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag/vectorstore"))
# Chroma 내 컬렉션 이름
COLLECTION_NAME = "subsidy_docs"
# 실제 참고 .txt 파일
SOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/subsidy_guide.txt"))

def ensure_vectorstore():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

    if not os.path.exists(PERSIST_DIR) or len(os.listdir(PERSIST_DIR)) == 0:
        print("📦 벡터 DB가 없어 생성을 시작합니다...")

        # 1. 문서 로드
        loader = TextLoader(SOURCE_PATH, encoding="utf-8")
        documents = loader.load()

        # 2. 문서 분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(documents)

        # 3. 벡터 DB 생성
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
        vectordb.persist()
        print("✅ 벡터 DB 생성 완료")
    else:
        print("📦 기존 벡터 DB를 불러옵니다...")

    # 4. DB 로딩
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})


# 5. 프롬프트 템플릿 구성
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

#  6. LLM 체인 구성 및 응답 생성
def build_chain():
    retriever = ensure_vectorstore()

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=800,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = get_subsidy_prompt()

    # LCEL 체인
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

# 답변 생성
_chain = build_chain()

def answer(question: str) -> str:
    if not question.strip():
        return "질문을 입력해주세요."

    try:
        return _chain.invoke({"question": question})
    except Exception as e:
        print(f"[❌ 오류 발생]: {e}")
        return "답변 중 오류가 발생했습니다."
