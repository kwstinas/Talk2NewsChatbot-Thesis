# Talk2News Chatbot

> A locally-hosted, retrieval-augmented news chatbot built on Llama 3.1, FAISS, and MongoDB. Crawls 16 Greek and international news sources, indexes articles semantically, and answers user queries in natural language with citations — entirely offline, no cloud APIs required.

Developed as part of a thesis project at the International Hellenic University, Department of Informatics, Computers and Telecommunications Engineering (2026).

---

## Overview

Talk2News combines automated news ingestion with a Retrieval-Augmented Generation (RAG) pipeline to let users ask questions about current events in plain English or Greek. Unlike cloud-based chatbots, the entire stack — crawler, vector store, and language model — runs on a single local machine, ensuring privacy, zero operational cost, and independence from external services.

---

## Key Features

### Intelligent Retrieval Pipeline
- **Hybrid Search**: Combines FAISS semantic similarity with BM25 lexical re-ranking for both conceptual and keyword-based precision.
- **Recency Scoring**: Time-weighted ranking prioritizes recent articles, critical for news applications where freshness matters.
- **MMR (Maximal Marginal Relevance)**: Eliminates redundancy across retrieved chunks to maximize information diversity within the LLM's context window.
- **Query Routing**: Classifies user intent (factual, analytical, follow-up, temporal) and enriches queries with detected entities before retrieval.

### Conversational Memory
- **Context-Aware Follow-ups**: A dedicated Conversation Manager tracks the last-used article and original query, enabling natural multi-turn conversations ("Tell me more about that", "What about Europe?").
- **Automatic Topic Detection**: Distinguishes between follow-up questions and new topics using lexical overlap heuristics and indicator phrases.

### Bilingual Support
- Automatic language detection (Greek/English) routes queries to dedicated prompt templates.
- Answers are generated in the same language as the question, preserving tone and formatting conventions.

### Multi-Article Synthesis
- For broad queries, the system selects multiple articles across diverse sources (max 2 per outlet) and synthesizes a unified response with per-source citations.
- Title deduplication prevents near-identical stories from dominating the results.

### Daily Digest
- A dedicated `/api/digest` endpoint generates a curated news summary on demand, pulling from multiple thematic categories over a configurable time window.

### Automated Ingestion
- Hourly RSS crawling from 16 curated Greek and international sources via APScheduler.
- MD5-based deduplication and MongoDB unique indexes prevent duplicate storage.
- Incremental FAISS updates add new embeddings without rebuilding the entire index.
- In-memory vectorstore with 30-minute auto-reload for near-real-time query responsiveness.

---
## Technology Stack

### Backend
- **Python 3.11+** — Primary language for all server-side logic
- **FastAPI** — Asynchronous REST API framework
- **Uvicorn** — ASGI server for production-grade async handling
- **APScheduler** — Background job scheduling for hourly crawls

### AI / ML
- **Llama 3.1 8B Instruct** (GGUF Q5_K_M quantization) — Local LLM inference
- **llama-cpp-python** — C++ bindings enabling CPU-only execution
- **FAISS** — Dense vector similarity search
- **sentence-transformers/all-mpnet-base-v2** — 768-dimensional embedding model
- **rank_bm25** — Lexical re-ranking algorithm

### Data Layer
- **MongoDB** — Document store for raw articles and metadata
- **PyMongo** — Official MongoDB driver for Python
- **FeedParser** — RSS/Atom feed parsing
- **BeautifulSoup4** — HTML cleaning and content extraction

### Frontend
- **React.js** (via CDN + Babel) — Component-based SPA
- **Fetch API** — Asynchronous HTTP communication
- **CSS Variables** — Dynamic dark/light theming
- **localStorage** — Client-side persistence for chat history and favorites

---

## Installation

### Prerequisites
- Python 3.11 or higher
- MongoDB running locally or accessible via URL
- At least 8 GB RAM (16 GB recommended)
- ~6 GB free disk space for the Llama model

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/kwstinas/Talk2NewsChatbot-Thesis.git
   cd Talk2NewsChatbot-Thesis
```

2. **Install Python dependencies**
```bash
   pip install -r requirements.txt
```

3. **Download the Llama 3.1 model**
   Due to size constraints, the model is not included. Download `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf` from Hugging Face and place it in the `models/` directory.

4. **Configure MongoDB**
   Set the `MONGO_URL` environment variable (defaults to `mongodb://localhost:27017/`):
```bash
   export MONGO_URL="mongodb://localhost:27017/"
```

5. **Start the server**
```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

6. **Open the frontend**
   Navigate to `http://localhost:8000` in your browser.

---
## Academic Context

This project was developed as a Bachelor's thesis under the supervision of **Dr. Stavros Vologiannidis**, Professor, International Hellenic University, Department of Informatics, Computers and Telecommunications Engineering, Serres campus.

**Author:** Dimitrios Kostinas
