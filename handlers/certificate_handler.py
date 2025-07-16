from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma

def answer(user_query):
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(persist_directory="./my_rag_db", embedding_function=embeddings)
    llm = OpenAI()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())

    result = qa.invoke({"query": user_query})
    print("\n🤖 [증명서 발급 챗봇 답변]")
    print(result["result"])