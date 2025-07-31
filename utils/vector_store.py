from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# 환경변수 로딩 (Google API 키)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# 임베딩 모델 설정
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# 벡터 DB 설정
VECTOR_DIR = "../my_rag_db"
COLLECTION_NAME = "admin_docs"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print("▶ BASE_DIR:", BASE_DIR)
print("▶ BASE_DIR contents:", os.listdir(BASE_DIR))

# 파일 경로
target_file = os.path.join(BASE_DIR, "training_handbook.txt")
if not os.path.exists(target_file):
    raise FileNotFoundError(f"❌ 파일 없음: {target_file}")

# 전체 문서 읽기
with open(target_file, encoding="utf-8") as f:
    full_text = f.read()

# 문단 단위로 분리하여 Document 객체 생성
sections = full_text.split("\n\n")
documents = [
    Document(page_content=section.strip(), metadata={"source": target_file})
    for section in sections if section.strip()
]

if not documents:
    raise ValueError("📂 문서에서 유효한 섹션을 찾지 못했습니다.")

# 청크화: 문단 기준 + 토큰 기준
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# 벡터 DB 저장
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_DIR
)

vectorstore.persist()
print("✅ 전체 문서 임베딩 및 Chroma 저장 완료")