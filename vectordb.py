import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPT_FILE = "cleaned_transcript.txt"

def load_chunks():
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split("\n\n")

def build_vector_db():
    print("🔄 Building FAISS in-memory vector DB (HuggingFace embeddings)...")

    chunks = load_chunks()

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_texts(chunks, embedder)
    print("✅ FAISS Vector DB ready.")
    return vectordb

db = build_vector_db()
