from langchain.chains import RetrievalQA
from search1 import llm,vectorstore
from fastapi import FastAPI
import uvicorn


app = FastAPI()


@app.get("/message")
def read_item(q: str = None):
    query = q
    vectorstore.similarity_search(
    query,  # our search query
    k= 1 # return 1 most relevant docs
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    a = qa.invoke(query)
    return a

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
