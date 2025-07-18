from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# 환경변수
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 임베딩 모델 (검색용)
embedding_model = OpenAIEmbeddings(
    openai_api_key=openai_key,
    model="text-embedding-3-small"
)

# 벡터 DB 로딩 -> /utils/vector_store_two.py를 통해 벡터 DB 최초 생성
VECTOR_DIR = "./utils/attendance_db"
COLLECTION_NAME = "leave_docs"

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_DIR,
    embedding_function=embedding_model
)

# OpenAI GPT 모델
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    openai_api_key=openai_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    return_source_documents=True
)

# 외부에서 호출할 함수
def answer(user_input: str) -> str:
    if not user_input:
        return "질문을 입력해주세요."

    try:
        result = qa_chain(user_input)

        # 디버깅용
        print("\n🔍 [DEBUG] 검색된 문서 수:", len(result["source_documents"]))
        for i, doc in enumerate(result["source_documents"]):
            print(f"\n📄 문서 {i+1}:\n{doc.page_content[:300]}")

        return str(result["result"])
    except Exception as e:
        print(f"[❌ 오류 발생]: {e}")
        return "답변 중 오류가 발생했습니다."
