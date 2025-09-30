# backend/app/chatbot/vectorstore.py
import os
import json
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ..utils_text import clean_html  

SAVE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_vectorstore_cache = None
_embeddings_cache = None


def _get_embeddings():
    """Επιστρέφει embeddings instance."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_cache


def load_vectorstore():
    """
    Φορτώνει το FAISS index από SAVE_PATH χρησιμοποιώντας τα ίδια embeddings.
    Κρατάει την ίδια ροή logs στα ελληνικά.
    """
    global _vectorstore_cache
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
        logger.info("Το vectorstore φορτώθηκε επιτυχώς!")
        print("Το vectorstore φορτώθηκε επιτυχώς!")
        return vs
    except Exception as e:
        logger.error(f"Σφάλμα κατά τη φόρτωση του vectorstore: {e}")
        return None


def similarity_search(query: str, k: int = 5):
    """
    Εκτελεί similarity search στο vectorstore.
    Επιστρέφει λίστα με (document, score).
    """
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
