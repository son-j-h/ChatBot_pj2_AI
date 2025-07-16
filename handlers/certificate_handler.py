from dotenv import load_dotenv

load_dotenv()

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

llm = OpenAI(temperature=0)


def load_vector_db():
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        persist_directory="C:/Users/user/IdeaVsProject/my_chatbot_project/my_rag_db",
        embedding_function=embeddings,
    )
    return vector_db


vector_db = load_vector_db()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())


def answer(user_input: str):
    # 모든 질문을 벡터 DB + LLM 질의응답으로 처리
    response = qa_chain.invoke({"query": user_input})
    print("🤖 증명서 발급 안내입니다:\n")
    print(response["result"])
