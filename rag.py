
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
def query_rag(question):
    result = qa.invoke({"query": question})  # <- use invoke instead of __call__
    answer = result["result"]
    sources = [doc.metadata["source"] for doc in result["source_documents"]]
    return answer, sources
