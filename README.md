# Talk2News Chatbot

A news-oriented chatbot built as part of an academic thesis project.  
The system crawls news articles from multiple Greek and international sources, stores them in MongoDB, and enables users to query the data conversationally through a Retrieval-Augmented Generation (RAG) pipeline powered by Llama 3.1.

---

##  Features

- **Automated Web Crawling**
  - Scheduled crawling from a curated list of news websites (Greek & international).
  - Extraction of full article content, titles, categories, and publication dates.
  - Duplicate detection and incremental updates to avoid redundant data.

- **Vector Search & RAG**
  - Article embeddings generated using `sentence-transformers/all-MiniLM-L6-v2`.
  - FAISS vectorstore for fast semantic search.
  - Hybrid ranking (similarity + recency + keyword/topic boost + BM25 reranking).
  - Ensures responses are based on a single, recent, and relevant article.

- **LLM Integration**
  - Runs locally with [llama.cpp](https://github.com/ggerganov/llama.cpp).
  - Configured with **Llama 3.1 8B Instruct (Q5_K_M quantization)**.
  - Strict system prompts enforce concise, factual answers drawn only from retrieved content.
  - Supports both English and Greek queries.

- **Backend**
  - FastAPI application serving REST endpoints.
  - APScheduler for automated hourly crawling.
  - Clear modular structure (`crawler`, `chatbot`, `api`).

- **CLI Demo**
  - Interactive script (`ask_chatbot.py`) for quick local testing without the API.

---
## Key Features That Make It Useful

**Bilingual Intelligence**
The system understands and responds in both English and Greek, making it equally useful for local Greek news and international coverage.

**Context-Aware Conversations**
Ask "Tell me about the new AI developments" and then follow up with "What about European regulations?" - the system maintains context naturally.

**Fresh Information**
With hourly crawling and smart updating, you're always getting information from the latest available articles rather than static knowledge.

**Multi-Source Synthesis**
When multiple outlets cover the same story, the system can identify different perspectives and provide a more comprehensive answer.

---
## Getting Started

The system is designed to be run locally, keeping your queries private and avoiding API costs. You'll need to provide your own Llama model file (due to size constraints), but everything else is included and ready to run..
