Talk2News Chatbot-Thesis


Overview


Talk2News Chatbot is a news chatbot designed to answer user questions related to current news articles. This project is based on Retrieval-Augmented Generation (RAG) architecture, which utilizes pre-trained machine learning models and information retrieval techniques to provide accurate, up-to-date answers. The chatbot gathers news articles from various sources, stores them in a database, and then leverages a powerful language model to generate answers based on those articles.

Objective
The goal of this project is to build a functional news chatbot capable of:

Collecting real-time news articles from multiple sources.

Storing and indexing these articles for easy retrieval.

Using a state-of-the-art language model (LLaMA 2) to answer user queries based on the articles.

Handling large datasets and models without exceeding GitHub's file upload limitations.

Key Features & Components
1. News Crawling & Database Integration
The chatbot collects news articles from several RSS feeds and stores them in a MongoDB database. This system periodically scrapes news websites and categorizes articles by topics like politics, economy, and technology.

Key Tools Used:
Feedparser: For scraping news articles from RSS feeds.

MongoDB: For storing the news articles.

2. Vector Search with FAISS
The chatbot uses FAISS (Facebook AI Similarity Search) to index and search articles efficiently. It splits the articles into smaller chunks and creates embeddings using the sentence-transformers model. These embeddings are stored in the FAISS vectorstore for fast similarity searches, enabling the chatbot to retrieve relevant articles for a given query.

Key Tools Used:
FAISS: For indexing and fast retrieval.

Sentence-Transformers: For generating embeddings of text.

3. LLaMA 2 Integration
The core language model used for generating answers is LLaMA 2 (7B parameters), a powerful generative language model. It uses the RAG approach to provide answers based on the retrieved documents from the database. The chatbot generates responses by combining relevant articles with the user’s query.

Key Tools Used:
LLaMA 2: For generating responses from retrieved documents.

LangChain: For integrating the LLaMA 2 model into the pipeline.

4. FastAPI Backend
The backend API is built with FastAPI, enabling users to interact with the chatbot through a RESTful interface. The FastAPI application supports user queries, retrieves the most relevant news articles, and generates answers using LLaMA 2.

Key Tools Used:
FastAPI: For building the backend and API.

Uvicorn: For serving the application.

What Has Been Built So Far
Completed:
RSS Crawler: Successfully collects articles from various news websites (e.g., Newsbeast, Naftemporiki, Greek Reporter, The Guardian, CNN, TechCrunch).

Database Integration: Stores news articles in a MongoDB database.

Vectorstore: The news articles are indexed using FAISS for fast retrieval.

LLaMA 2 Model Integration: LLaMA 2 has been integrated for answering queries based on the indexed articles.

FastAPI Backend: A RESTful API is built using FastAPI, allowing interaction with the chatbot.

In Progress:
Frontend Integration: The frontend user interface for querying the chatbot is under development. The integration will allow users to interact with the chatbot easily through a web interface.

Improved Answer Generation: Work is ongoing to fine-tune the language model responses and ensure they are coherent and relevant to user queries.

Technologies & Tools Used
Programming Languages: Python

Web Framework: FastAPI

Database: MongoDB

Text Search: FAISS

Machine Learning Models: LLaMA 2 (7B Q4_K_M GGUF)

Embedding Model: Sentence-Transformers (all-MiniLM-L6-v2)

Task Scheduling: APScheduler

Version Control: Git & GitHub

Environment Management: Virtualenv (venv)

Files Excluded from the GitHub Repository
Due to GitHub's file size limitations, the following files have been excluded from the repository:

ML models (e.g., .gguf, .bin, .h5, .pt, .pth).

Large datasets (e.g., .zip, .npy, .pkl files larger than 1MB).

Virtual environment files (venv/ directory).

Python bytecode files (__pycache__/, .pyc).

These files were excluded intentionally, as they exceed GitHub's file size limits (2GB per file). The necessary models and large datasets are stored locally or can be downloaded separately.
