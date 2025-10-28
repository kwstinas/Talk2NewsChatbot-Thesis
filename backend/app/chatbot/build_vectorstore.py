# backend/app/chatbot/build_vectorstore.py
import os
import shutil
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from dateutil import parser

from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---- Ρυθμίσεις (όπως τα έχεις ήδη) ----
MONGO_URL = "mongodb://172.25.240.1:27017/"
DATABASE_NAME = "news_database"
COLLECTION_NAME = "articles"
SAVE_PATH = "faiss_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Mongo helpers -----------------
def _collection():
    client = MongoClient(MONGO_URL)
    db = client[DATABASE_NAME]
    return db[COLLECTION_NAME]

def _load_articles_all() -> List[dict]:
    col = _collection()
    arts = list(col.find())
    if not arts:
        raise ValueError("Δεν βρέθηκαν άρθρα στη βάση δεδομένων!")
    logger.info(f"Φορτώθηκαν {len(arts)} άρθρα.")
    return arts

def _parse_flexible_date(date_str: str) -> datetime:
    """
    Parse διαφορετικών date formats με intelligent fallback.
    """
    if not date_str:
        return None
        
    # Προσπάθησε ISO format πρώτα (πιο γρήγορο)
    try:
        if 'T' in date_str:  # ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        pass
        
    # Fallback σε dateutil parser (πιο flexible αλλά πιο αργό)
    try:
        dt = parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except:
        logger.warning(f"⚠️ Could not parse date: {date_str}")
        return None

def _load_articles_since(iso_utc: str) -> List[dict]:
    """
    Φέρνει άρθρα με fetched_at >= iso_utc (πιο αξιόπιστο από published_date).
    """
    col = _collection()
    
    # Χρησιμοποίησε fetched_at που είναι πάντα ISO format και αξιόπιστο
    arts = list(col.find({"fetched_at": {"$gte": iso_utc}}).sort("fetched_at", -1).limit(200))
    
    # Fallback: αν δεν βρεθεί τίποτα με fetched_at, δοκίμασε manual filtering
    if not arts:
        logger.info(f"Δεν βρέθηκαν άρθρα με fetched_at, δοκιμάζω manual filtering...")
        
        # Πάρε τα τελευταία 1000 άρθρα για performance
        all_recent_arts = list(col.find().sort("_id", -1).limit(1000))
        
        try:
            cutoff_dt = datetime.fromisoformat(iso_utc.replace('Z', '+00:00'))
        except:
            cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=24)
        
        arts = []
        for art in all_recent_arts:
            pub_date_str = art.get("published_date", "")
            if not pub_date_str:
                continue
                
            pub_dt = _parse_flexible_date(pub_date_str)
            if pub_dt and pub_dt >= cutoff_dt:
                arts.append(art)
        
        logger.info(f"Βρέθηκαν {len(arts)} άρθρα με manual filtering.")
    else:
        logger.info(f"Φορτώθηκαν {len(arts)} άρθρα από {iso_utc} και μετά.")
    
    return arts

# ----------------- Chunking -----------------
def _create_chunks(articles: List[dict]) -> Tuple[List[str], List[dict]]:
    texts, metadatas = [], []
    for a in articles:
        content = (a.get("content") or "").strip()
        title = a.get("title", "Άγνωστος Τίτλος")
        link = a.get("link", "")
        published_date = a.get("published_date", "")
        category = a.get("category", "General")

        if not content:
            continue

        combined_text = f"{title}\n\n{content}"
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(combined_text), step):
            chunk = combined_text[i : i + CHUNK_SIZE]
            if chunk.strip():
                texts.append(chunk)
                metadatas.append(
                    {
                        "title": title,
                        "link": link,
                        "published_date": published_date,
                        "category": category,
                    }
                )
    logger.info(f"Δημιουργήθηκαν {len(texts)} chunks.")
    return texts, metadatas

# ----------------- Meta helpers -----------------
def _meta_path() -> str:
    return os.path.join(SAVE_PATH, "meta.json")

def _read_meta() -> dict:
    try:
        with open(_meta_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_meta(meta: dict):
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ----------------- Full build -----------------
def build_vectorstore_full():
    """
    Καθαρό rebuild του FAISS index από ΟΛΑ τα άρθρα.
    """
    logger.info("Ξεκινά η διαδικασία FULL δημιουργίας του FAISS vectorstore...")

    articles = _load_articles_all()
    texts, metadatas = _create_chunks(articles)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Καθαρό rebuild
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        logger.info("🧹 Διαγράφηκε το προηγούμενο FAISS index.")

    vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vs.save_local(SAVE_PATH)

    # meta
    try:
        dim = len(embeddings.embed_query("dimension-probe"))
    except Exception:
        dim = None

    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": dim,
        "chunks": len(texts),
        "last_built_iso": _utc_now_iso(),  # timestamp τελευταίου build
    }
    _write_meta(meta)

    logger.info(f"✅ Ολοκληρώθηκε το FULL FAISS vectorstore με {len(texts)} chunks!")
    logger.info(f"ℹ️ Embedding model: {EMBEDDING_MODEL} | dim={dim}")

# ----------------- Incremental update -----------------
def incremental_update_vectorstore(hours: int = 24):
    """
    Προσθέτει ΜΟΝΟ τα νέα άρθρα των τελευταίων `hours` (default 24 ώρες)
    στο υπάρχον FAISS index. Αν δεν υπάρχει index, κάνει full build.
    """
    try:
        if not os.path.exists(SAVE_PATH):
            logger.warning("Δεν υπάρχει FAISS index — γίνεται FULL build.")
            build_vectorstore_full()
            return

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Διαβάζουμε meta (για πληροφοριακούς λόγους)
        meta = _read_meta()

        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff_dt.isoformat()

        # Φέρε νέα άρθρα από Mongo
        articles = _load_articles_since(cutoff_iso)
        if not articles:
            logger.info("🔎 Δεν βρέθηκαν νέα άρθρα για incremental ενημέρωση.")
            return

        texts, metadatas = _create_chunks(articles)
        if not texts:
            logger.info("🔎 Δεν προέκυψαν νέα chunks για incremental ενημέρωση.")
            return

        # Φόρτωσε τον υπάρχον FAISS και κάνε merge
        logger.info("Φόρτωση υπάρχοντος FAISS index για merge...")
        base_vs = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)

        logger.info(f"Δημιουργία προσωρινού index με {len(texts)} νέα chunks...")
        tmp_vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

        logger.info("Συγχώνευση (merge) προσωρινού index στο βασικό...")
        base_vs.merge_from(tmp_vs)
        base_vs.save_local(SAVE_PATH)

        # Ενημέρωση meta
        new_chunks_total = (meta.get("chunks") or 0) + len(texts)
        try:
            dim = meta.get("embedding_dim") or len(embeddings.embed_query("dimension-probe"))
        except Exception:
            dim = meta.get("embedding_dim")

        meta.update(
            {
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": dim,
                "chunks": new_chunks_total,
                "last_built_iso": _utc_now_iso(),
            }
        )
        _write_meta(meta)

        logger.info(f"✅ Incremental ενημέρωση ολοκληρώθηκε. Προστέθηκαν {len(texts)} νέα chunks. Σύνολο ~{new_chunks_total}.")

    except Exception as e:
        logger.error(f"❌ Σφάλμα incremental update: {e}")
        raise

# Προηγούμενο entrypoint για συμβατότητα
def build_vectorstore():
    build_vectorstore_full()

if __name__ == "__main__":
    # CLI: τρέχει full build αν καλέσεις απευθείας το script
    build_vectorstore_full()