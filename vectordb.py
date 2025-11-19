# vectordb.py (Streamlit Cloud safe version)

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = "models/embedding-001"

TRANSCRIPT_FILE = "cleaned_transcript.txt"

def load_chunks():
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split("\n\n")


def build_vector_db():
    print("🔄 Building Chroma vector DB...")

    chunks = load_chunks()

    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        api_key=GOOGLE_API_KEY
    )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedder,
        collection_name="its_transcript",
        persist_directory="./chroma_db"
    )

    vectordb.persist()
    print("✅ Chroma DB ready.")
    return vectordb


# Build when imported
db = build_vector_db()
