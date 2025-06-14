# backend/app/chatbot/build_vectorstore.py

import os
import shutil
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----- Ρυθμίσεις -----
MONGO_URL = 'mongodb://172.25.240.1:27017/'
DATABASE_NAME = 'news_database'
COLLECTION_NAME = 'articles'
SAVE_PATH = "faiss_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----- MongoDB σύνδεση -----
def connect_to_mongo():
    client = MongoClient(MONGO_URL)
    db = client[DATABASE_NAME]
    return db[COLLECTION_NAME]

# ----- Φόρτωση άρθρων -----
def load_articles():
    collection = connect_to_mongo()
    articles = list(collection.find())

    if not articles:
        raise ValueError(" Δεν βρέθηκαν άρθρα στη βάση δεδομένων!")

    print(f" Φορτώθηκαν {len(articles)} άρθρα για επεξεργασία.")
    return articles

# ----- Δημιουργία chunks -----
def create_chunks(articles):
    texts = []
    metadatas = []

    for article in articles:
        content = article.get('content', '').strip()
        title = article.get('title', 'Άγνωστος Τίτλος')
        link = article.get('link', '')
        published_date = article.get('published_date', '')
        category = article.get('category', 'General')

        if not content:
            continue

        combined_text = f"{title}\n\n{content}"

        for i in range(0, len(combined_text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = combined_text[i:i + CHUNK_SIZE]
            if chunk.strip():
                texts.append(chunk)
                metadatas.append({
                    "title": title,
                    "link": link,
                    "published_date": published_date,
                    "category": category
                })

    print(f" Δημιουργήθηκαν {len(texts)} chunks για embeddings.")
    return texts, metadatas

# ----- Δημιουργία και αποθήκευση FAISS vectorstore -----
def build_vectorstore():
    print(" Ξεκινά η διαδικασία δημιουργίας του FAISS vectorstore...")

    articles = load_articles()
    texts, metadatas = create_chunks(articles)

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        print("🧹 Διαγράφηκε το προηγούμενο FAISS index.")

    vectorstore = FAISS.from_texts(texts, embedder, metadatas=metadatas)
    vectorstore.save_local(SAVE_PATH)

    print(f"✅ Ολοκληρώθηκε η δημιουργία FAISS vectorstore με {len(texts)} chunks!")

# ----- Εκκίνηση -----
if __name__ == "__main__":
    build_vectorstore()
