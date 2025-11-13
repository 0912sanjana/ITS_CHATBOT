🚀 Dhamm AI ITS Chatbot
Conversational RAG Assistant for Knowledge-Driven Learning

The Dhamm AI ITS Chatbot is an intelligent, retrieval-augmented conversational tutor built using LangChain, Google Gemini, and Groq LLaMA 3.3.
It provides strict, transcript-driven answers, ensuring high precision, explainability, and Bloom’s Taxonomy–aligned reasoning.

📘 Overview

This chatbot integrates:
Semantic retrieval (Chroma Vector DB)
Strict context enforcement
Dual-LLM inference (Gemini + Groq failover)
A modular Flask backend
Adaptive and domain-specific Q&A for ITS systems

🧱 Core Tech Stack
Component	Technology
Embeddings	Google Generative AI (models/text-embedding-004)
Primary LLM	Gemini 2.5 Pro
Fallback LLM	Groq Llama 3.3-70B Versatile
Backend	Flask
Vector Store	ChromaDB
Retrieval	LangChain Community
Orchestration	LangChain-Chroma
Environment	.env (API keys ignored via .gitignore)

🟧 Key Highlights (2025 Update)
Category	Improvements
🔄 Dual Model	Gemini primary + Groq fallback (auto failover on quota/429)
📚 Strict Transcript Context	Answers ONLY from cleaned_transcript.txt; out-of-scope → polite refusal
🧠 Vector DB	Rebuilt vectordb.py with safe chunking + improved embeddings
⚙️ Error Handling	Clear JSON responses, handled exceptions
💾 Environment Safety	.env securely ignored
🧩 Modular Design	Separate vector builder + API router
🤯 RAG Pipeline	Better chunking, retrieval consistency
🌐 Cross-Platform	Works on Windows, macOS, Linux & Azure


🧩 Architecture Overview
Transcript (cleaned_transcript.txt)
          │
          ▼
RecursiveCharacterTextSplitter (Chunking)
          │
          ▼
Google Embeddings (text-embedding-004)
          │
          ▼
Chroma Vector Store (vectordb/)
          │
          ▼
Retriever (Top-K = 4)
          │
          ▼
LLM Layer:
   - Gemini 2.5 Pro (primary)
   - Groq LLaMA 3.3-70B (fallback)
          │
          ▼
Flask API Response (/api/chat)


⚒️ Setup Guide
1️⃣ Clone the Repository
git clone https://github.com/0912sanjana/ITS_CHATBOT.git
cd ITS_CHATBOT

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🔐 Environment Variables

Create a .env file:

GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here


⚠️ Never commit .env → It is correctly ignored using .gitignore.

🧠 Build Vector Database

Before running the chatbot:
python vectordb.py

This will:
Load cleaned_transcript.txt
Split text into chunks
Create embeddings
Save vector DB into vectordb/

Expected output:

✅ VectorDB built and persisted.
📦 Documents indexed: 1
🔢 Chroma internal count: 120

🗣️ Run the Chatbot Server
python app.py


Access UI/API at:
👉 http://localhost:8000

⚙️ API Endpoints
1. Health Check

GET /

Response:

{
  "status": "ok",
  "message": "Dhamm AI backend is running."
}

2. Chat Endpoint

POST /api/chat

Example:

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain the DKT model"}'


Response:

{
  "answer": "Here is the answer strictly from transcript..."
}


Out-of-scope example:

{
  "answer": "Sorry, this question is outside the scope of the provided transcript."
}

3. Question Generator

POST /api/generate-questions

{
  "course_outcome": "Understand AI models",
  "bloom_level": "apply"
}


Uses Groq → fallback Gemini.

📂 Vector Store Structure
File	Description
cleaned_transcript.txt	RAG data source
vectordb/	Chroma embeddings
vectordb.py	Vector DB builder

☁️ Deployment Options
▶ Gunicorn
gunicorn app:app --bind 0.0.0.0:8000 --workers 4 --worker-class gevent --timeout 120

▶ Azure App Service
Upload project folder
Configure GOOGLE_API_KEY and GROQ_API_KEY in App Settings
Enable CORS
Use startup command: gunicorn app:app

🧪 Testing
In-scope:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Hybrid KG-RAG?"}'

Out-of-scope:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Who won FIFA World Cup 2022?"}'

🔒 Security Notes

Never push .env or API keys
Use Azure Key Vault / Secrets Manager in production
Rotate expired tokens regularly
Use HTTPS for deployed API

🤝 Contributing

PRs welcome!
Fork the repo
Create a feature branch
Submit Pull Request

🧾 License

MIT © 2025 Dhamm AI LLP
Empowering adaptive, intelligent tutoring systems.
