from fastapi import FastAPI, HTTPException
from rag_engine import get_answer
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()  # Load MONGO_URL

app = FastAPI(title="College Enquiry Chatbot")

# MongoDB connection
mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    raise ValueError("MONGO_URL not found in .env")
client = MongoClient(mongo_url)
db = client["college_chatbot"]
collection = db["questions"]  # Cache collection

@app.post("/ask")
def ask_question(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 1️⃣ Check MongoDB first
    cached = collection.find_one({"question": question})
    if cached:
        return {"answer": cached["answer"]}

    # 2️⃣ If not found, query RAG engine
    answer = get_answer(question)

    # 3️⃣ Store the new Q&A in MongoDB
    collection.insert_one({"question": question, "answer": answer})

    return {"answer": answer}
