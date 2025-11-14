import os
import streamlit as st
from dotenv import load_dotenv

# LangChain + Vector DB
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

# =============================
# Environment Variables
# =============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY is missing in .env or Secrets!")
    st.stop()

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing in .env or Secrets!")
    st.stop()

# =============================
# RAG Config
# =============================
EMBED_MODEL = "models/text-embedding-004"
PRIMARY_GEMINI = "gemini-2.5-pro"
VECTORDB_DIR = "vectordb"
COLLECTION_NAME = "chroma"
TOP_K = 4

# =============================
# Load Vector DB
# =============================
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        api_key=GOOGLE_API_KEY
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORDB_DIR,
        embedding_function=embeddings
    )

vectorstore = load_vectorstore()

# =============================
# LLM Selection with Fallback
# =============================
def call_llm(prompt):
    # Try Gemini first
    try:
        model = ChatGoogleGenerativeAI(
            model=PRIMARY_GEMINI,
            api_key=GOOGLE_API_KEY
        )
        res = model.invoke(prompt)
        return res.content
    except Exception as e:
        # FALLBACK → Groq LLaMA 3.3 70B
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=GROQ_API_KEY
            )
            res = llm.invoke(prompt)
            return res.content
        except:
            return "❌ LLM Error: Unable to generate response."

# =============================
# Strict RAG Answer
# =============================
def rag_answer(question):
    docs = vectorstore.similarity_search(question, k=TOP_K)

    if not docs:
        return "Sorry, this question is outside the scope of the provided transcript."

    context = "\n\n".join([d.page_content for d in docs])
    REFUSAL = "Sorry, this question is outside the scope of the provided transcript."

    prompt = f"""
You are a Strict RAG chatbot. 
Use ONLY the context below to answer the user.
If answer is not in context → reply exactly:

"{REFUSAL}"

Context:
{context}

Question: {question}

Answer:
"""

    response = call_llm(prompt)
    if REFUSAL in response:
        return REFUSAL

    return response

# =============================
# STREAMLIT UI
# =============================

st.set_page_config(page_title="Dhamm AI – ITS RAG Chatbot", layout="wide")

st.title("🤖 Dhamm AI – ITS ITS Chatbot")
st.caption("Strict RAG chatbot answering only from the ITS transcript. Out-of-scope questions are politely refused.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chats
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_message = st.chat_input("Ask your ITS question…")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.write(user_message)

    with st.chat_message("assistant"):
        answer = rag_answer(user_message)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
