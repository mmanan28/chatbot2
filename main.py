from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
import uvicorn
from search1 import get_greeting_reply, llm, vectorstore

app = FastAPI()

# Define request model for POST API

class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def print_message(request: MessageRequest):
    try:
        query = request.message
        # Perform similarity search
        search_results = vectorstore.similarity_search(query, k=1)  # Adjust k as needed

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Check for greeting or blog reply
        reply = get_greeting_reply(query)
        if reply:
            return {"reply": reply}

        # If not a greeting, perform retrieval-based QA
        modified_query = query + " in points"
        response = qa.invoke(modified_query)
        if 'result' in response:
            return {"result": response["result"]}
        else:
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)