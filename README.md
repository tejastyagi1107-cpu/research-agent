# ResearchAI — CS Paper Assistant

An AI-powered web application that helps students understand dense computer science research papers — algorithms, data structures (DSA), and AI/ML literature — using **IBM watsonx.ai** (Llama 3.3 70B Instruct) and a RAG pipeline with FAISS vector search.

---

## ✨ Features

| Feature | Description |
|---|---|
| **📤 Upload Paper** | Upload a PDF or paste raw text; auto-chunked and embedded into a local FAISS index |
| **📋 Summary View** | Structured summary: contributions, algorithms, complexity analysis, experiments |
| **💬 Chat / Q&A** | Ask natural-language questions; answers grounded in the paper with citations |
| **💡 Concept Explainer** | Step-by-step explanations of any CS/AI/ML concept with real-world analogies |
| **🔁 Conversational** | Multi-turn chat with history context, beginner-friendly tone |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser (SPA)                    │
│   Home · Upload · Chat · Summary · Concept Explainer│
└───────────────────┬─────────────────────────────────┘
                    │ REST API (JSON)
┌───────────────────▼─────────────────────────────────┐
│              Flask Application                       │
│  /upload/   /chat/ask   /summary/generate            │
│  /explain/concept                                    │
└────────┬──────────────────────────┬─────────────────┘
         │                          │
┌────────▼────────┐       ┌─────────▼─────────────────┐
│   RAG Pipeline  │       │    IBM watsonx.ai          │
│  pdfplumber     │       │  meta-llama/llama-3-3-70b  │
│  sentence-trans │       │  (via ibm-watsonx-ai SDK)  │
│  FAISS index    │       └───────────────────────────-┘
└─────────────────┘
```

---

## 🚀 Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd research-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your IBM watsonx.ai credentials:

```env
WATSONX_API_KEY=your_api_key_here
WATSONX_URL=your_watsonx_url_here
WATSONX_PROJECT_ID=your_project_id_here
```

> **Get credentials:** Log in to [IBM Cloud](https://cloud.ibm.com) → IAM → API Keys. Your project ID is in watsonx.ai project settings.

### 3. Run the application

```bash
python run.py
```

Open [http://localhost:5001](http://localhost:5001) in your browser.

---

## 🌐 Deployment

### Heroku

```bash
heroku create your-app-name
heroku config:set WATSONX_API_KEY=<key>
heroku config:set WATSONX_URL=https://au-syd.ml.cloud.ibm.com
heroku config:set WATSONX_PROJECT_ID=<project_id>
heroku config:set WATSONX_MODEL_ID=meta-llama/llama-3-3-70b-instruct
heroku config:set SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
git push heroku main
```

> **Note:** FAISS indexes are stored on disk. On Heroku's ephemeral filesystem, re-upload is needed after dyno restart. For persistence, consider mounting a volume or using a managed vector DB.

### IBM Cloud Code Engine

```bash
ibmcloud ce application create \
  --name research-agent \
  --image <your-registry>/research-agent:latest \
  --env WATSONX_API_KEY=<key> \
  --env WATSONX_URL=https://au-syd.ml.cloud.ibm.com \
  --env WATSONX_PROJECT_ID=<id> \
  --env WATSONX_MODEL_ID=meta-llama/llama-3-3-70b-instruct \
  --env SECRET_KEY=<secret>
```

---

## 📁 Project Structure

```
research-agent/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── watsonx_client.py    # IBM watsonx.ai SDK wrapper (singleton)
│   ├── rag.py               # Chunking, FAISS indexing, retrieval
│   ├── pdf_utils.py         # PDF extraction (pdfplumber)
│   ├── prompts.py           # Prompt templates + LLM calls
│   └── routes/
│       ├── main.py          # Home page
│       ├── upload.py        # POST /upload/
│       ├── chat.py          # POST /chat/ask
│       ├── summary.py       # POST /summary/generate
│       └── explain.py       # POST /explain/concept
├── templates/
│   └── index.html           # Single-page app shell
├── static/
│   ├── css/style.css        # Dark theme UI
│   └── js/app.js            # Vanilla JS frontend logic
├── data/
│   └── vector_stores/       # FAISS indexes (auto-created)
├── run.py                   # Dev server entry point
├── Procfile                 # Heroku/gunicorn config
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `WATSONX_API_KEY` | ✅ | IBM Cloud API key |
| `WATSONX_URL` | ✅ | watsonx.ai service URL (region-specific) |
| `WATSONX_PROJECT_ID` | ✅ | watsonx.ai project ID |
| `WATSONX_MODEL_ID` | ✅ | Model ID (default: `meta-llama/llama-3-3-70b-instruct`) |
| `SECRET_KEY` | ✅ | Flask session signing key |
| `VECTOR_STORE_DIR` | ❌ | FAISS index storage path (default: `data/vector_stores`) |

---

## 🧠 How RAG Works

1. **Ingest** — The uploaded PDF/text is extracted and split into 500-word overlapping chunks (100-word overlap).
2. **Embed** — Each chunk is embedded using `all-MiniLM-L6-v2` (sentence-transformers, runs locally).
3. **Index** — Embeddings are stored in a FAISS `IndexFlatL2` index keyed by a content hash.
4. **Retrieve** — For each user query, the top-k most semantically similar chunks are retrieved.
5. **Generate** — Retrieved chunks are injected into a structured prompt sent to watsonx.ai for grounded generation.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `ibm-watsonx-ai` | Official IBM SDK for watsonx.ai model inference |
| `faiss-cpu` | Efficient similarity search for dense vectors |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `pdfplumber` | PDF text extraction |
| `flask` | Web framework |
| `gunicorn` | Production WSGI server |

---

## 🛡 Security Notes

- Never commit `.env` to version control.
- Rotate your `SECRET_KEY` before deploying to production.
- The API key in `.env.example` is a placeholder — replace it with your own.
