from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# ✅ 1. 임베딩 모델 설정
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# ✅ 2. 문서 기반 RAG 벡터 DB (읽기 전용)
rag_vectordb = Chroma(
    persist_directory="../my_rag_db",   # 벡터 DB 경로 주의!
    collection_name="admin_docs",
    embedding_function=embedding_model
)

# ✅ 3. 실시간 대화 저장용 벡터 DB (쓰기 전용)
memory_vectordb = Chroma(
    persist_directory="memory_db",      # 별도 저장소
    collection_name="chat_history",
    embedding_function=embedding_model
)

def retrieve_context(user_input, student_id=None):
    """
    ✅ 문서 기반 RAG DB에서 유사한 문서를 검색합니다.
    실시간 대화 벡터는 사용하지 않음.
    """
    try:
        results = rag_vectordb.similarity_search_with_score(user_input, k=3)
        return [doc for doc, score in results if score > 0.8]
    except Exception as e:
        print(f"❌ RAG 검색 오류: {e}")
        return []

def save_chat_to_vectorstore(user_input, response, student_id=None):
    """
    ✨ 실시간 대화 내용을 벡터화하여 저장합니다.
    검색에는 사용하지 않음.
    """
    try:
        content = f"User: {user_input}\nAssistant: {response}"
        metadata = {"student_id": student_id} if student_id else {}
        doc = Document(page_content=content, metadata=metadata)

        memory_vectordb.add_documents([doc])
        memory_vectordb.persist()
        print("✅ 실시간 대화 저장 완료")
    except Exception as e:
        print(f"❌ 대화 저장 실패: {e}")