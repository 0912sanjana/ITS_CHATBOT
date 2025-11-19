# FINAL FAST EMBEDDING BUILDER (NO HF MODELS)
import os
import chromadb

TRANSCRIPT = "cleaned_transcript.txt"

client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_collection("its_transcript")
    client.delete_collection("its_transcript")
except:
    pass

collection = client.create_collection("its_transcript")

chunks = []

with open(TRANSCRIPT, "r", encoding="utf-8") as f:
    text = f.read()

# Simple fast chunking
for part in text.split("\n\n"):
    cleaned = part.strip()
    if len(cleaned) > 20:
        chunks.append(cleaned)

# Store chunks (no embeddings required!)
for idx, c in enumerate(chunks):
    collection.add(
        ids=[f"doc_{idx}"],
        documents=[c]
    )

print("🔥 FAST VectorDB created with NO embeddings.")
