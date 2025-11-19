import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from langchain_groq import ChatGroq

# ---------------------------------------------------
# FLASK SETUP
# ---------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------
# ENV
# ---------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

# ---------------------------------------------------
# LOAD GROQ LLM (cached)
# ---------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
)

# ---------------------------------------------------
# LOAD CHROMA — NO EMBEDDINGS (FAST MODE)
# ---------------------------------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("its_transcript")  # same as vectordb.py

REFUSAL = "Sorry, this question is outside the ITS transcript."


# ---------------------------------------------------
# FAST RETRIEVAL
# ---------------------------------------------------
def fast_retrieve(query):
    try:
        res = collection.query(
            query_texts=[query],
            n_results=3
        )
        return res["documents"][0]
    except:
        return []


# ---------------------------------------------------
# STRICT RAG ANSWER
# ---------------------------------------------------
def rag_answer(question: str, mode: str = "short"):
    docs = fast_retrieve(question)

    if not docs:
        return REFUSAL

    context = "\n\n".join(docs)

    style = (
        "Explain briefly in 3–4 lines."
        if mode == "short"
        else "Explain in detail with proper structure."
    )

    prompt = f"""
You are Dhamm AI – a strict ITS RAG chatbot.
ONLY answer using the transcript context below.
If answer is missing → reply exactly: "{REFUSAL}"

Context:
{context}

Question: {question}

Rules:
- NO outside knowledge
- NO hallucination
- {style}

Answer:
"""

    try:
        res = llm.invoke(prompt)
        return res.content.strip()
    except Exception as e:
        return f"Server Error: {e}"


# ---------------------------------------------------
# API ROUTES
# ---------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "short").lower()

    if not question:
        return jsonify({"error": "Missing question"}), 400

    answer = rag_answer(question, mode)
    return jsonify({"answer": answer})


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Dhamm AI backend running"})


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
