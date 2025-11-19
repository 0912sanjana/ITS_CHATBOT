import os
import streamlit as st
from dotenv import load_dotenv
import chromadb
from langchain_groq import ChatGroq

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------
st.set_page_config(
    page_title="Dhamm AI – ITS RAG Chatbot (Fast Version)",
    page_icon="🧠",
    layout="wide",
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------
# 1. CACHE LLM – loads ONCE
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
    )

llm = init_llm()

# ---------------------------------------------------
# 2. CACHE CHROMA – loads ONCE
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    # collection name must match vectordb.py
    return client.get_collection("its_transcript")

collection = init_chroma()

# ---------------------------------------------------
# 3. FAST TOP-K RETRIEVAL
# ---------------------------------------------------
def fast_retrieve(query: str, k: int = 4):
    """
    Retrieve top-k chunks from Chroma for a query.
    Returns a list of text chunks.
    """
    try:
        result = collection.query(
            query_texts=[query],
            n_results=k
        )
        docs = result.get("documents", [[]])[0]
        return [d for d in docs if d.strip()]
    except Exception:
        return []


# ---------------------------------------------------
# 4. STRICT RAG ANSWER (SHORT / DETAILED)
# ---------------------------------------------------
def answer_query(user_input: str, mode: str = "short") -> str:
    """
    mode = 'short' or 'detailed'
    """
    docs = fast_retrieve(user_input, k=5)

    if not docs:
        return "Sorry, this question is outside the ITS transcript."

    context = "\n\n".join(docs)

    if mode == "short":
        style_instruction = (
            "Answer in clear English, in 3–4 concise lines. "
            "Focus only on the key idea."
        )
    else:
        style_instruction = (
            "Give a detailed explanation in good English. "
            "Use 1–3 short paragraphs and bullet points where helpful. "
            "Keep the tone professional but easy to understand."
        )

    prompt = f"""
You are **Dhamm AI**, an expert tutor for the Dhamm.AI PKM-integrated Intelligent Tutoring System.

You MUST answer using only the information in the transcript context below.
If the answer is truly not present in the context, you must say politely:
"Sorry, this question is outside the ITS transcript."

Do NOT mention the words "transcript" or "context" in your answer.
Do NOT invent architecture or features that are not in the context.

{style_instruction}

=== START OF CONTEXT ===
{context}
=== END OF CONTEXT ===

User question: {user_input}
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return "⚠️ The server is busy right now. Please try again in a moment."


# ---------------------------------------------------
# 5. SESSION STATE FOR CHAT HISTORY
# ---------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role": "user"|"bot", "text": str}


# ---------------------------------------------------
# 6. SIDEBAR – ANSWER STYLE & INFO
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛 Answer Style")
    style_choice = st.radio(
        "Choose:",
        ["Short (3–4 lines)", "Detailed (explain with depth)"],
        index=1,
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []
        st.success("Chat cleared.")

    st.markdown("---")
    st.markdown("**Model:** Groq LLaMA 3.1 8B")
    st.markdown("**Embeddings:** MiniLM-L6-v2 (Local, super-fast)")
    st.markdown("**Database:** ChromaDB (local persistent)")

    st.markdown("---")
    st.markdown(
        "💡 *This bot answers only from your ITS transcript: "
        "PKM + DKT + BNKG + architecture + roadmap, etc.*"
    )

# ---------------------------------------------------
# 7. HEADER
# ---------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 Dhamm AI – ITS RAG Chatbot (Fast Version)</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Strict RAG chatbot — answers only from your cleaned ITS transcript. "
    "Perfect for viva prep, demos, and faculty review.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------
# 8. CHAT HISTORY UI
# ---------------------------------------------------
for item in st.session_state.chat:
    role, text = item["role"], item["text"]
    if role == "user":
        bg = "#FFF5F5"
        label = "🧑 You"
    else:
        bg = "#F0FFF4"
        label = "🤖 Bot"

    st.markdown(
        f"""
        <div style="background:{bg};padding:12px 16px;
                    border-radius:10px;margin:6px 0;">
            <b>{label}:</b> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------
# 9. USER INPUT (FAST)
# ---------------------------------------------------
user_q = st.chat_input("Ask your ITS / PKM / DKT / architecture question…")

if user_q:
    # add user message
    st.session_state.chat.append({"role": "user", "text": user_q})

    # choose mode
    mode = "short" if style_choice.startswith("Short") else "detailed"

    with st.spinner("Thinking over the ITS transcript…"):
        ans = answer_query(user_q, mode)

    # add bot answer
    st.session_state.chat.append({"role": "bot", "text": ans})

    # re-render with new messages
    st.rerun()
