from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# 환경 변수 로딩
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 임베딩 모델
embedding_model = OpenAIEmbeddings(
    openai_api_key=openai_key,
    model="text-embedding-3-small"
)

# 벡터 DB 경로와 컬렉션
VECTOR_DIR = "../my_rag_db"
COLLECTION_NAME = "chat_history"

# 벡터 DB 불러오기
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_DIR,
    embedding_function=embedding_model
)

# 청크 분할기
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 🟢 실시간 대화 저장 함수
def save_chat_to_vectorstore(user_input, bot_response, student_id="default"):
    chat = f"User: {user_input}\nBot: {bot_response}"
    doc = Document(page_content=chat, metadata={"source": "chat", "student_id": student_id})
    chunks = splitter.split_documents([doc])
    vectorstore.add_documents(chunks)

# 🔍 쿼리 유사 검색 함수
def retrieve_context(query, k=3, student_id=None):
    if student_id:
        return vectorstore.similarity_search(query, k=k, filter={"student_id": student_id})
    return vectorstore.similarity_search(query, k=k)
