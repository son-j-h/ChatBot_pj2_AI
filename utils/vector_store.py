from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"API 키 설정 여부: {'있음' if api_key else '없음'}")


def load_and_chunk_text(filepath: str) -> list[Document]:
    """텍스트 파일을 불러와 섹션별로 Document 청크 생성"""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sections = raw_text.split("\n\n")
    docs = []

    for section in sections:
        lines = section.strip().split("\n")
        if not lines or len(lines[0].strip()) == 0:
            continue

        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        category = title.split(". ")[-1] if ". " in title else title

        doc = Document(
            page_content=f"[{category}]\n{content}",
            metadata={"category": category, "source_file": os.path.basename(filepath)},
        )
        docs.append(doc)

    return docs


# 두 개의 파일을 청크화
docs1 = load_and_chunk_text("training_handbook.txt")
docs2 = load_and_chunk_text("trainee_admin_attendance_guide.txt")

# 전체 문서 통합
all_docs = docs1 + docs2

# 벡터 DB 생성
embeddings = OpenAIEmbeddings()
vector_db = Chroma.from_documents(all_docs, embeddings, persist_directory="./my_rag_db")

print(f"총 {len(all_docs)}개의 문서 청크가 벡터 DB에 저장되었습니다.")
