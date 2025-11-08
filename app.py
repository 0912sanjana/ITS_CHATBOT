import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Vector store & models
# If you migrated to langchain-chroma, switch import to: from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ------------- Flask -------------
app = Flask(__name__, static_url_path='')
CORS(app)

# ------------- ENV / Keys -------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is missing! Set GOOGLE_API_KEY in your .env")
if not GROQ_API_KEY:
    raise ValueError("Groq API Key is missing! Set GROQ_API_KEY in your .env")

# ------------- Config -------------
EMBEDDING_MODEL = "models/text-embedding-004"
VECTORDB_DIR = "vectordb"          # must match your vectordb.py
COLLECTION_NAME = "chroma"         # must match your vectordb.py
K = 4                               # retrieve top-k chunks
REFUSAL = "Sorry, this question is outside the scope of the provided transcript."
PRIMARY_GEMINI = os.getenv("CHAT_PRIMARY_GEMINI", "gemini-2.5-pro")   # or "gemini-1.5-flash"

# ------------- Vector store loader -------------
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY
    )
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORDB_DIR,
        embedding_function=embeddings
    )
    return vs

vectorstore = get_vectorstore()

# ------------- LLM factory + fallback -------------
def make_llm(provider: str, temperature: float = 0.2):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=PRIMARY_GEMINI,
            api_key=GOOGLE_API_KEY,
            temperature=temperature
        )
    elif provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",     # current Groq recommended 70B general model
            api_key=GROQ_API_KEY,
            temperature=temperature
        )
    else:
        # default to gemini
        return ChatGoogleGenerativeAI(
            model=PRIMARY_GEMINI,
            api_key=GOOGLE_API_KEY,
            temperature=temperature
        )

def llm_invoke_with_fallback(prompt: str, temperature: float = 0.2) -> str:
    """
    Try Gemini -> on quota/429/model issues or provider errors, fallback to Groq.
    Returns text content, or a string starting with "GenerationError" on hard failure.
    """
    # Try Gemini first
    try:
        g = make_llm("gemini", temperature)
        res = g.invoke(prompt)
        return getattr(res, "content", None) or str(res)
    except Exception as e:
        msg = str(e)
        # Detect rate/quota/decom issues and fallback
        if ("429" in msg) or ("quota" in msg.lower()) or ("rate" in msg.lower()) or ("model_decommissioned" in msg.lower()):
            try:
                v = make_llm("groq", temperature)
                res = v.invoke(prompt)
                return getattr(res, "content", None) or str(res)
            except Exception as e2:
                return f"GenerationError after fallback: {type(e2).__name__}: {e2}"
        return f"GenerationError: {type(e).__name__}: {e}"

# ------------- Strict RAG answer -------------
def answer_from_transcript(question: str, show_chunks: bool = False):
    # Retrieve chunks (no threshold initially; we hard-refuse if none)
    try:
        docs = vectorstore.similarity_search(question, k=K)
    except Exception as e:
        return {"error": f"RetrievalError: {e}"}, 500

    if not docs:
        return {"answer": REFUSAL, "context": ""}, 200

    context = "\n\n".join([d.page_content for d in docs])

    prompt = (
        "You are a strict retrieval assistant.\n"
        "ONLY use the context below to answer the question.\n"
        f"If the context does not answer the question, reply exactly:\n\"{REFUSAL}\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    content = llm_invoke_with_fallback(prompt, temperature=0.2)

    if content.startswith("GenerationError"):
        return {"error": content}, 500

    if (not content.strip()) or (REFUSAL in content):
        return {"answer": REFUSAL}, 200


    result = {"answer": content.strip()}
    if show_chunks:
     	result["context"] = context
    return result, 200

# ------------- Routes -------------
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    show_chunks = data.get("show_chunks", False)

    if not question:
        return jsonify({"error": "Missing question"}), 400

    payload, status = answer_from_transcript(question, show_chunks=show_chunks)
    return jsonify(payload), status


@app.route("/api/generate-questions", methods=["POST"])
def generate_questions():
    """
    Defaults to Groq (to save Gemini quota), with fallback to Gemini if Groq fails.
    """
    data = request.get_json(silent=True) or {}
    course_outcome = data.get("course_outcome")
    bloom_level = data.get("bloom_level")

    if not course_outcome or not bloom_level:
        return jsonify({"error": "Missing course_outcome or bloom_level"}), 400

    prompt = (
        "You are an expert question generator.\n"
        "Based on the following course outcome and Bloom level, generate:\n"
        "- One objective MCQ with 4 options (A-D)\n"
        "- One short answer subjective question\n\n"
        f"Course Outcome: {course_outcome}\n"
        f"Bloom Level: {bloom_level}\n\n"
        "Format your response like this:\n"
        "Objective Question:\n"
        "...\n"
        "A. ...\n"
        "B. ...\n"
        "C. ...\n"
        "D. ...\n\n"
        "Short Answer Question:\n"
        "..."
    )

    # Try Groq first for this endpoint
    try:
        q_llm = make_llm("groq", temperature=0.2)
        res = q_llm.invoke(prompt)
        content = getattr(res, "content", None) or str(res)
    except Exception as e:
        # Fallback to Gemini if Groq fails
        content = llm_invoke_with_fallback(prompt, temperature=0.2)
        if content.startswith("GenerationError"):
            return jsonify({"error": content}), 500

    lines = [ln for ln in content.strip().split("\n")]
    options = [line.strip() for line in lines if line.strip().startswith(tuple("ABCD"))]
    subjective_start = next((i for i, l in enumerate(lines) if "Short Answer Question" in l), -1)
    subjective = (
        lines[subjective_start + 1].strip()
        if subjective_start >= 0 and (subjective_start + 1) < len(lines)
        else "N/A"
    )

    return jsonify({
        "bloom_level": bloom_level,
        "course_outcome": course_outcome,
        "questions": {
            "objective": {"question": "Here are the generated questions:", "options": options},
            "subjective": subjective
        },
        "raw_text": content
    })

@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Dhamm AI backend is running."})

@app.route("/test")
def test_ui():
    return open("test.html", encoding="utf-8").read()

# ------------- Run -------------
if __name__ == "__main__":
    # Use 8000 to avoid macOS AirPlay conflicts
    app.run(host="0.0.0.0", port=8000, debug=True)
