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
## Architecture
[Web Sources / RSS Feeds]
│
▼
[Crawler - feedparser]
│
▼
[MongoDB Database]
│
▼
[Vectorstore - FAISS + HuggingFace Embeddings]
│
▼
[RAG Pipeline]
│
▼
[LLM - Llama 3.1 (via llama.cpp)]
│
▼
[FastAPI Backend / CLI Tool]
│
▼
[User Interface]

##  Next Steps

While the current version of the project is fully functional, there are several improvements planned:

- Develop a simple **frontend interface** (React-based chatbot) for easier user interaction.  
- Optimize **retrieval and ranking** for even more accurate answers. 
- Improve the **LLM integration** for more concise and factual responses.
