import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load FAISS DB from vectordb.py
from vectordb import db as vectordb

# LLM (Groq)
from langchain_groq import ChatGroq

# ---------------------------------------
# Flask Setup
# ---------------------------------------
app = Flask(__name__)
CORS(app)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

# LLM instance (fast)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
)

# ---------------------------------------
# RAG Answer Function
# ---------------------------------------
def answer_query(question, show_chunks=False):
    try:
        docs = vectordb.similarity_search(question, k=4)
    except Exception as e:
        return {"error": f"RetrievalError: {e}"}, 500

    if not docs:
        return {"answer": "Sorry, this question is outside the transcript."}, 200

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a strict RAG chatbot.
Use ONLY the context below to answer.

Context:
{context}

Question: {question}

If answer is not found in context, say:
"Sorry, this question is outside the transcript."
"""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        return {"error": f"LLMError: {e}"}, 500

    data = {"answer": answer}
    if show_chunks:
        data["context"] = context

    return data, 200

# ---------------------------------------
# Routes
# ---------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    show_chunks = data.get("show_chunks", False)

    if not question:
        return jsonify({"error": "Missing question"}), 400

    result, status = answer_query(question, show_chunks)
    return jsonify(result), status


@app.route("/")
def home():
    return jsonify({"status": "ok", "backend": "FAISS-based RAG"})


# ---------------------------------------
# Run
# ---------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

