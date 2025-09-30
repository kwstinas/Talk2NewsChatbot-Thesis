# backend/app/chatbot/build_vectorstore.py
import os
import shutil
import json
import logging
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

MONGO_URL = "mongodb://172.25.240.1:27017/"
DATABASE_NAME = "news_database"
COLLECTION_NAME = "articles"
SAVE_PATH = "faiss_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_to_mongo():
    client = MongoClient(MONGO_URL)
    db = client[DATABASE_NAME]
    return db[COLLECTION_NAME]


def load_articles():
    collection = connect_to_mongo()
    articles = list(collection.find())
    if not articles:
        raise ValueError("Δεν βρέθηκαν άρθρα στη βάση δεδομένων!")
    logger.info(f"Φορτώθηκαν {len(articles)} άρθρα.")
    return articles


def create_chunks(articles):
    texts, metadatas = [], []
    for article in articles:
        content = (article.get("content") or "").strip()
        title = article.get("title", "Άγνωστος Τίτλος")
        link = article.get("link", "")
        published_date = article.get("published_date", "")
        category = article.get("category", "General")

        if not content:
            continue

        combined_text = f"{title}\n\n{content}"
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(combined_text), step):
            chunk = combined_text[i: i + CHUNK_SIZE]
            if chunk.strip():
                texts.append(chunk)
                metadatas.append({
                    "title": title,
                    "link": link,
                    "published_date": published_date,
                    "category": category
                })
    logger.info(f"Δημιουργήθηκαν {len(texts)} chunks.")
    return texts, metadatas


def build_vectorstore():
    logger.info("Ξεκινά η διαδικασία δημιουργίας του FAISS vectorstore...")

    articles = load_articles()
    texts, metadatas = create_chunks(articles)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # καθαρό rebuild
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        logger.info("🧹 Διαγράφηκε το προηγούμενο FAISS index.")

    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(SAVE_PATH)

    # γράφουμε ένα meta.json για να αποφεύγουμε “διάσταση-λάθος”
    try:
        dim = len(embeddings.embed_query("dimension-probe"))
    except Exception:
        dim = None

    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": dim,
        "chunks": len(texts)
    }
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Ολοκληρώθηκε η δημιουργία FAISS vectorstore με {len(texts)} chunks!")
    logger.info(f"ℹ️ Embedding model: {EMBEDDING_MODEL} | dim={dim}")


if __name__ == "__main__":
    build_vectorstore()
