import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing!")
    st.stop()

# =============================
# Load transcript
# =============================
TRANSCRIPT_PATH = "cleaned_transcript.txt"

if not os.path.exists(TRANSCRIPT_PATH):
    st.error("❌ cleaned_transcript.txt missing!")
    st.stop()

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = f.read().split("\n\n")

# =============================
# Tiny Fast Embedding Model
# =============================
def embed(text):
    # VERY FAST HASHING EMBEDDING
    vector = np.zeros(300)
    for i, word in enumerate(text.split()):
        vector[i % 300] += len(word)
    return vector

embeddings = np.array([embed(chunk) for chunk in transcript])

# =============================
# Similarity Search
# =============================
def search(query, k=4):
    q_vec = embed(query)
    scores = np.dot(embeddings, q_vec)
    idx = scores.argsort()[-k:][::-1]
    return [transcript[i] for i in idx]


# =============================
# Load Groq LLM
# =============================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0
)

# =============================
# Strict RAG Answer
# =============================
REFUSE = "Sorry, this question is outside the scope of the provided transcript."

def rag_answer(question):
    docs = search(question)
    context = "\n\n".join(docs)

    prompt = f"""
STRICT RAG MODE — Answer using ONLY the context.
If the answer is not in the context, reply with:

"{REFUSE}"

Context:
{context}

Question: {question}

Answer:
"""

    result = llm.invoke(prompt)
    return result.content.strip()


# =============================
# UI
# =============================
st.title("🤖 Dhamm AI – ITS RAG Chatbot (Fast Version)")

user_q = st.chat_input("Ask your ITS question…")

if user_q:
    with st.chat_message("user"):
        st.write(user_q)

    answer = rag_answer(user_q)

    with st.chat_message("assistant"):
        st.write(answer)
