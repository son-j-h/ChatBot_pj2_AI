from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.helpers import load_text ,load_few_shot_examples
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()


vector_db = Chroma(
    persist_directory="./my_rag_db", # 이 자리값 수정 해야함
    embedding_function=embeddings
)


# def load_text(filepath):
#     with open(filepath, "r", encoding="utf-8") as f:
#         return f.read()

# def load_few_shot_examples(filepath):
#     with open(filepath, "r", encoding="utf-8") as f:
#         return f.read()

# # txt에서 직접 원본 데이터 읽기
# full_text = load_text("utils/vacation_data.txt")   # 또는 "data/vacation_data.txt"     # 얘도 벡터db 통합불러오기 하면서 필요없어짐

CHUNK_SIZE = 300
CHUNK_OVERLAP = 80
SEARCH_K = 3
PROMPT_STYLE = ("공식적인 어조로, 반드시 존댓말로, 번호로 요약해 답변하라. "
    "※ 단, '존댓말로 하라', '번호로 요약하라' 등 이 지침 문구를 답변에 직접 적지 마세요."
    "'사용자 질문'에만 집중해서 답변하라. 이전 대화 내용은 참고만 하고, 현재 질문에 맞지 않는 정보는 절대 포함하지마라"
)
TEMPERATURE = 0.05
MAX_TOKENS = 500

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=CHUNK_SIZE,
#     chunk_overlap=CHUNK_OVERLAP
# )
# chunks = splitter.split_text(full_text)
# docs = [Document(page_content=c, metadata={"idx": i}) for i, c in enumerate(chunks)]
# embeddings = OpenAIEmbeddings()
# vector_db = Chroma.from_documents(
#     docs,
#     embeddings,
#     persist_directory="./vacation_db_chunksize"      ## 벡터db 불러오기 1번만 하기 위해 삭제할 코드들 주석처리
# )
llm = OpenAI(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
shot_examples = load_few_shot_examples("utils/vacation_shot.txt")

def answer(query: str) -> str:
    try:
        results = vector_db.similarity_search(query, k=SEARCH_K)
        context = "\n\n".join([doc.page_content for doc in results])

        prompt = f"""
        [참고 데이터]
        {context}

        [예시 Q/A]
        {shot_examples}

        [지침] {PROMPT_STYLE}
        사용자 질문: {query}
        위 정보/예시/샷만 참고해서 답변해줘.
        """
        answer = llm.invoke(prompt)
        return answer.strip()
    except Exception as e:
        print(f"[핸들러 오류] {e}")
        return "죄송합니다. 답변 처리 중 오류가 발생했습니다."
