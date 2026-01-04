from fastapi import FastAPI
from rag_engine import get_answer

app = FastAPI()

@app.post("/ask")
def ask_question(question: str):
    answer = get_answer(question)
    return {"answer": answer}
