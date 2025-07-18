# 휴가/조퇴/병가/공가 관련해서 대답할 수 있게끔 벡터DB 파일 수정

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# OpenAI 임베딩 모델 사용
embedding_model = OpenAIEmbeddings(
    openai_api_key=openai_key,
    model="text-embedding-3-small"
)

# 벡터 DB가 저장될 폴더 생성
VECTOR_DIR = "./attendance_db"
COLLECTION_NAME = "leave_docs"

doc_paths = [
    "attendance_guide.txt"
]

# 벡터 DB화 시킬 문서가 존재하지 않을 경우
all_documents = []
for path in doc_paths:
    if not os.path.exists(path):
        print(f"❌ 파일 없음: {path}")
        continue
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()
    all_documents.extend(documents)

if not all_documents:
    raise ValueError("📂 임베딩할 문서를 찾을 수 없습니다.")

# 청크화
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(all_documents)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_DIR
)
vectorstore.persist()
print("✅ 문서 임베딩 완료 및 Chroma 저장")