#  Dhamm AI ITS Chatbot

### Conversational RAG Assistant for Knowledge-Driven Learning

Dhamm AI Chatbot is an intelligent, retrieval-augmented chat application built on **LangChain**, **Google Gemini**, and **Groq Llama 3.3**.
It is designed to deliver **context-aware answers** strictly from your uploaded transcript — ensuring precision, transparency, and explainability — with **Bloom’s Taxonomy**-aligned reasoning.

---

##  Overview

This chatbot combines **semantic retrieval**, **strict context control**, and **multi-model inference** (Gemini + Groq fallback) to power real-time, adaptive tutoring and domain-specific Q&A.

**Core stack:**

* **LangChain Community + LangChain-Chroma**
* **Google Generative AI Embeddings**
* **Gemini 2.5 Pro** (primary LLM)
* **Groq Llama 3.3-70B Versatile** (automatic fallback)
* **Flask API Backend**
* **Chroma Vector Database**

---

##  Key Highlights (2025 Update)

| Category                     | Improvements                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| 🔄 Dual Model                | Gemini (primary) + Groq (fallback) with automatic fail-over on quota/429 errors             |
| 📚 Strict Transcript Context | Chatbot answers only from `cleaned_transcript.txt`; out-of-scope queries politely refused   |
| 🧠 Vector DB                 | Rebuilt `vectordb.py` with safe rebuild, better chunking, and explicit Google embedding key |
| ⚙️ Error Handling            | Graceful exceptions + clear JSON messages                                                   |
| 💾 Environment Safety        | `.env` excluded via `.gitignore`; secure key handling validated                             |
| 🌐 Portability               | Works on macOS, Linux, and Azure App Service                                                |
| 🧩 Modular Design            | Separate vector builder (`vectordb.py`) + Flask API (`app.py`)                              |
| 🤯 RAG Pipeline              | Improved retrieval thresholds and chunk context resolution                                  |

---

## 🧩 Architecture Overview

1. **Transcript → Chunks** → via `RecursiveCharacterTextSplitter`
2. **Chunks → Embeddings** → Google `models/text-embedding-004`
3. **Chroma Vector Store** → stored locally in `vectordb/`
4. **RAG Chain** → retrieves top-K context per query
5. **Gemini LLM → Groq Fallback** → contextual response generation
6. **Flask API** → serves REST endpoints for chat and question generation

---

## 🧮 Setup Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/dibyacharyaAI/ITS_chatbot.git
cd ITS_chatbot
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

> 🛡️ **Do NOT commit `.env`** — it’s ignored via `.gitignore`.

---

## 🧠 Initialize Vector Database

Before running the chatbot:

```bash
python3 vectordb.py
```

This will:

* Load `cleaned_transcript.txt`
* Split into semantic chunks
* Generate embeddings (Google)
* Persist to `vectordb/`

Expected output:

```
✅ VectorDB built and persisted.
📦 Documents indexed: 1 | Chunks indexed: 13
🔢 Chroma internal count: 13
```

---

## 🗣️ Run the Chatbot

### Development Mode

```bash
source venv/bin/activate
python3 app.py
```

### Access

Visit → [http://localhost:8000](http://localhost:8000)

Health check:

```bash
curl http://localhost:8000/
```

---

## ⚙️ API Endpoints

### 1. Health Check

**GET** `/`

```json
{ "status": "ok", "message": "Dhamm AI backend is running." }
```

### 2. Chat

**POST** `/api/chat`

```json
{ "question": "Which schedule of the Indian Constitution mentions powers of Panchayats?" }
```

Returns:

```json
{
  "answer": "The Eleventh Schedule of the Constitution lists the powers of Panchayats.",
  "context": "..."
}
```

> 🧩 **Important JSON Syntax Note:** When testing with curl, always use lowercase `true` and `false` in JSON — Python-style `True` / `False` will break parsing.
>
> Example:
>
> ```bash
> curl -X POST http://localhost:8000/api/chat \
>   -H "Content-Type: application/json" \
>   -d '{
>     "question": "Which schedule of the Indian Constitution mentions powers of Panchayats?",
>     "show_chunks": true
>   }'
> ```
>
> ✅ Works fine.
>
> ❌ This will fail:
>
> ```bash
> "show_chunks": True
> ```

> Out-of-scope questions get: `{ "answer": "Sorry, this question is outside the scope of the provided transcript." }`

### 3. Generate Questions

**POST** `/api/generate-questions`

```json
{
  "course_outcome": "Understand soil mechanics",
  "bloom_level": "apply"
}
```

* Uses Groq by default → Gemini fallback

---

## 🧩 Vector Store Structure

| File                     | Description                       |
| ------------------------ | --------------------------------- |
| `cleaned_transcript.txt` | Raw source text for embedding     |
| `vectordb/`              | Persisted Chroma collection       |
| `vectordb.py`            | Script to build/rebuild the index |

---

## 🧱 Deployment Options

### ▶ Gunicorn (Local/Server)

```bash
gunicorn app:app --bind 0.0.0.0:8000 --workers 4 --worker-class gevent --timeout 120
```

### ▶ Azure App Service

* Upload project directory
* Configure App Settings:

  * `FLASK_ENV=production`
  * `GOOGLE_API_KEY` & `GROQ_API_KEY`
* Enable CORS for frontend domain

### ▶ React Frontend (Optional)

Connect via `axios` to `/api/chat` (see frontend section in original documentation).

---

## 🧩 Configuration Parameters

| Key               | Description               | Default                                 |
| ----------------- | ------------------------- | --------------------------------------- |
| `EMBEDDING_MODEL` | Google embedding model    | `"models/text-embedding-004"`           |
| `VECTORDB_DIR`    | Vector database directory | `"vectordb"`                            |
| `COLLECTION_NAME` | Chroma collection name    | `"chroma"`                              |
| `K`               | Top-K retrieval chunks    | 4                                       |
| `PRIMARY_GEMINI`  | Gemini model              | `"gemini-2.5-pro"` / `gemini-1.5-flash` |

---

## 🧪 Testing

```bash
# Chat (in-scope)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Which schedule of the Indian Constitution mentions powers of Panchayats?"}'

# Out-of-scope
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Who won the FIFA World Cup 2022?"}'
```

---

## 🦯 Troubleshooting

| Issue                 | Fix                                                                            |
| --------------------- | ------------------------------------------------------------------------------ |
| ❌ Quota 429           | Gemini quota hit → auto fallback to Groq                                       |
| ❌ VectorDB missing    | Rebuild with `python3 vectordb.py`                                             |
| ❌ Port in use         | Edit `app.run(..., port=8000)`                                                 |
| ❌ No response         | Verify .env keys + logs (`tail -f app.log`)                                    |
| ⚠️ LangChain warnings | `pip install -U langchain langchain-core langchain-community langchain-chroma` |

---

## 🔒 Security Notes

* Never commit `.env` or API keys.
* Use Azure Key Vault or environment variables for production secrets.
* Always rotate expired tokens (Google & Groq).

---

## 🤝 Contributing

Contributions welcome! Fork the repo → create a branch → submit a Pull Request.
For feature discussions or bug reports, open a GitHub Issue.

---

## 🧾 License

Licensed under the **MIT License**.
© 2025 **Dhamm AI LLP** — Building Adaptive Learning Intelligence for Education.
