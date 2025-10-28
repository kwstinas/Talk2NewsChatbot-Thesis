# backend/app/chatbot/vectorstore.py
import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ..utils_text import clean_html  

SAVE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Auto-reload settings
AUTO_RELOAD_INTERVAL = timedelta(minutes=30)  # Reload every 30 minutes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_vectorstore_cache = None
_embeddings_cache = None
_last_reload_time = None

def _get_embeddings():
    """Επιστρέφει embeddings instance."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings_cache

def _should_reload_vectorstore() -> bool:
    """Ελέγχει αν πρέπει να γίνει reload του vectorstore."""
    global _last_reload_time
    
    if _vectorstore_cache is None:
        return True
        
    if _last_reload_time is None:
        return True
        
    # Reload αν έχουν περάσει AUTO_RELOAD_INTERVAL λεπτά
    time_since_reload = datetime.now(timezone.utc) - _last_reload_time
    return time_since_reload > AUTO_RELOAD_INTERVAL

def _force_reload_vectorstore():
    """Αναγκάζει το reload του vectorstore στο επόμενο load."""
    global _vectorstore_cache, _last_reload_time
    _vectorstore_cache = None
    _last_reload_time = None
    logger.info("🔄 Vectorstore reload scheduled for next request")

def load_vectorstore(force_reload: bool = False):
    """
    Φορτώνει το FAISS index με auto-reload capability.
    """
    global _vectorstore_cache, _last_reload_time
    
    # Έλεγχος αν χρειάζεται reload
    if force_reload or _should_reload_vectorstore():
        _vectorstore_cache = None
        logger.info("🔄 Auto-reloading vectorstore...")
    
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    if not os.path.exists(SAVE_PATH):
        logger.warning(f"Το FAISS vectorstore δεν υπάρχει στο {SAVE_PATH}.")
        return None

    embeddings = _get_embeddings()

    # Προαιρετικός έλεγχος για mismatch embedding model
    meta_path = os.path.join(SAVE_PATH, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            built_model = meta.get("embedding_model")
            if built_model and built_model != EMBEDDING_MODEL:
                logger.warning(
                    f"⚠️ Το FAISS χτίστηκε με '{built_model}', αλλά τώρα ζητάς '{EMBEDDING_MODEL}'. "
                    f"Προτείνεται rebuild του index (τρέξε το build_vectorstore)."
                )
        except Exception as e:
            logger.warning(f"⚠️ Αδυναμία ανάγνωσης meta.json: {e}")

    try:
        logger.info(f"Φόρτωση του FAISS vectorstore από το {SAVE_PATH}.")
        vs = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
        _vectorstore_cache = vs
        _last_reload_time = datetime.now(timezone.utc)
        
        # Πληροφορίες για το νέο vectorstore
        doc_count = len(vs.docstore._dict)
        logger.info(f"✅ Το vectorstore φορτώθηκε επιτυχώς! ({doc_count} documents)")
        return vs
    except Exception as e:
        logger.error(f"Σφάλμα κατά τη φόρτωση του vectorstore: {e}")
        return None

def similarity_search(query: str, k: int = 5):
    """
    Εκτελεί similarity search με auto-reload check.
    """
    # Πάντα φόρτωσε το τελευταίο vectorstore πριν την αναζήτηση
    vs = load_vectorstore()
    if not vs:
        return []

    # Ασφαλής απολύμανση UTF-8 (κόβει surrogates/χαλασμένα bytes)
    if not isinstance(query, str):
        query = str(query or "")
    query = query.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Καθαρισμός HTML + trimming
    try:
        query_clean = clean_html(query).strip()
    except Exception:
        # Αν κολλήσει το BeautifulSoup με περίεργο input
        logging.exception("clean_html failed on query")
        query_clean = (query or "").strip()

    # Απλός κόφτης θορύβου / κενών
    if not query_clean or len(query_clean) < 2:
        logging.warning("similarity_search: άδειο ή πολύ μικρό query μετά τον καθαρισμό.")
        return []

    # Προληπτικό κόψιμο υπερβολικά μεγάλων queries (πχ paste dump)
    if len(query_clean) > 512:
        query_clean = query_clean[:512]

    logger.info("🔍 Εκτελείται similarity search για query='%s'...", query_clean[:60])

    try:
        return vs.similarity_search_with_score(query_clean, k=k)
    except Exception:
        # Δείξε πλήρες traceback για να βρούμε ρίζα (αντί για κενό μήνυμα)
        logger.exception("Σφάλμα στο similarity search")
        return []

# Νέα function για manual reload
def reload_vectorstore():
    """Αναγκάζει το reload του vectorstore."""
    _force_reload_vectorstore()
    return load_vectorstore(force_reload=True)

def get_vectorstore_info() -> dict:
    """Επιστρέφει πληροφορίες για το τρέχον vectorstore."""
    vs = load_vectorstore()
    if not vs:
        return {"status": "not_loaded"}
    
    try:
        meta_path = os.path.join(SAVE_PATH, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        
        reload_info = ""
        if _last_reload_time:
            time_since = datetime.now(timezone.utc) - _last_reload_time
            reload_info = f"{int(time_since.total_seconds() / 60)} minutes ago"
        
        return {
            "status": "loaded",
            "documents_count": len(vs.docstore._dict),
            "embedding_model": meta.get("embedding_model", "unknown"),
            "chunks_count": meta.get("chunks", 0),
            "last_built": meta.get("last_built_iso", "unknown"),
            "last_reload": reload_info
        }
    except Exception as e:
        logger.error(f"Error getting vectorstore info: {e}")
        return {"status": "error", "error": str(e)}