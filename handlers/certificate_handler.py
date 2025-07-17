from dotenv import load_dotenv
import os

load_dotenv()

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

llm = OpenAI(temperature=0)

embeddings = OpenAIEmbeddings()
vector_db = Chroma(
    persist_directory = os.path.join(os.getcwd(), "my_rag_db"),
    embedding_function=embeddings,
)

# 검색기와 LLM 체인 생성
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # 상위 3개 문서 검색
llm = OpenAI(temperature=0)  # 온도 0은 답변 안정성↑

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def answer(user_input: str) -> str:
    if not user_input.strip():
        return "질문을 입력해 주세요."

    try:
        # 벡터 DB에서 유사 문서 찾고, LLM으로 답변 생성
        response = qa_chain.run(user_input)
        print("🤖 증명서 발급 안내입니다:\n", repr(response))
        if not response.strip():
            return "죄송하지만, 답변을 찾을 수 없습니다."
        return response

    except Exception as e:
        print(f"답변 생성 중 오류 발생: {e}")
        return "답변을 생성하는 중 오류가 발생했습니다."
