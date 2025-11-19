# 🌸 Dhamm AI — ITS RAG Chatbot  
### **(Fast + Strict Version)**  

A strict **Retrieval-Augmented Generation (RAG)** chatbot that answers only from the **ITS cleaned transcript**.

Designed for:

- Viva preparation  
- Faculty review  
- Demonstrations  
- Smart learning support  

This chatbot guarantees **zero hallucination** by restricting answers strictly to retrieved transcript chunks.

---

## ⭐ Features

### 🔍 **Strict Transcript-Only Answers**
- The bot uses only the transcript to answer.  
- If a question is outside the transcript → **politely refuses**.

### ⚡ **Fast FAISS In-Memory Vector DB**
- Ultra-fast retrieval.  
- Completely **safe for Streamlit Cloud**.  
- No `.sqlite`, `.bin`, `.hnsw` files required.

### 🧠 **Gemini + Groq Dual LLM Support**
- Primary: **Gemini 1.5 / 2.0**  
- Backup: **Groq LLaMA-3.1 8B / 70B**  
- Automatic fallback included.

---

## 🚀 How It Works

1. Transcript is split into clean text chunks.  
2. Fast **FAISS** vector embeddings created at runtime.  
3. Top-K relevant chunks retrieved using semantic search.  
4. Gemini/Groq generates an answer **ONLY** from transcript context.  
5. If no relevant context is found → **refuses safely**.

---

## 📁 Project Structure

ITS_CHATBOT/
│
├── app.py # Flask backend API
├── streamlit_app.py # Streamlit UI
├── vectordb.py # FAISS vector builder
├── cleaned_transcript.txt
├── requirements.txt
├── README.md
└── .gitignore


---

## 🔒 .gitignore Included

Vector DB / Embeddings

vectordb/
chroma_db/
.sqlite
*.bin
*.hnsw
Cache
pycache/
Keys
.env
*.log


---

## 👩‍💻 Contribution
Maintained with 💛 by **Sanjana**.

---

## 📬 Support
For issues, feature requests or improvements → open an Issue in GitHub.

---

DONE ✔️

---

### Want me to auto-commit this to GitHub with correct formatting?  
Just say **“commit it”**.


