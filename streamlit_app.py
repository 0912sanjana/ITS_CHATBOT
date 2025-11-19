import streamlit as st
from vectordb import db
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="Dhamm AI – ITS RAG Chatbot (Fast Version)",
    layout="wide",
)

st.title("🧠 Dhamm AI — ITS RAG Chatbot (Fast Version)")
st.caption("Strict RAG chatbot — answers only from your cleaned ITS transcript.")

# LOCAL HF MODEL
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# RETRIEVAL
# ---------------------------
def retrieve_chunks(query):
    return db.similarity_search(query, k=4)

# ---------------------------
# STRICT ANSWERING
# ---------------------------
def strict_answer(query):
    docs = retrieve_chunks(query)

    if not docs:
        return "❌ Sorry, no matching content found in your transcript."

    context = "\n\n".join([d.page_content for d in docs])

    return f"""
📘 **Answer based ONLY on transcript:**

{context}
"""

# ---------------------------
# UI
# ---------------------------
st.write("### 🔍 Ask your ITS / PKM / DKT question:")

query = st.text_input("Enter question here")

if query:
    with st.spinner("Searching transcript…"):
        response = strict_answer(query)
    st.success(response)
