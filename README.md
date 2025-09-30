# 📰 Talk2News Chatbot

**Talk2News** is an interactive chatbot that provides fresh, reliable answers based on recent news articles.  
It combines **web crawling, semantic search, and local LLM inference (LLaMA 3)** to deliver up-to-date, contextual responses.
It is built as part of a final-year thesis, focusing on **Retrieval-Augmented Generation (RAG)** with local LLM inference.

The system fetches articles from multiple news sources, indexes them into a vector store, and answers user questions strictly based on the **most relevant and recent article**.

---

##  Features Implemented

- **Web Crawling & RSS Feeds**
  - Automated crawler fetches articles from multiple news sites.
  - Metadata extraction (title, date, category, author).

- **Database**
  - MongoDB for storing full articles with metadata.
  - Duplicate detection using hash keys.

- **Vector Store & Retrieval**
  - HuggingFace embeddings (`all-mpnet-base-v2`).
  - FAISS vector index for similarity search.
  - Hybrid retrieval: similarity + recency boost + BM25 re-ranking.
  - Selection of a **single best article** per query.

- **LLM Integration**
  - Local inference with **LLaMA 3** (quantized GGUF).
  - Prompting designed to return **concise, fact-based answers**.
  - Extractive snippet mechanism when available.

- **Frontend**
  - React-based chat interface (served without build tools, via Babel & CDN).
  - Custom **logo and branding**.
  - Persistent chat history (localStorage).

---

## Tech Stack

  Backend: Python, FastAPI, MongoDB

  Retrieval: FAISS, BM25 re-ranking

  Embeddings: HuggingFace all-mpnet-base-v2

  LLM: LLaMA 3 (GGUF, via llama-cpp-python)

  Frontend: React (served via Babel, no build step)

  ## Author

  Dimitris Kostinas
  Final-year Computer Science thesis project.
  Focus areas: RAG pipelines, LLM integration and Web Crawling.
