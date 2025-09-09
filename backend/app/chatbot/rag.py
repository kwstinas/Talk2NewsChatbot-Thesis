# backend/app/chatbot/rag.py
from datetime import datetime, timedelta, timezone
from dateutil import parser
import logging
import math
import re
import re as _re
import os
import re as _re_tok
from time import perf_counter
from dotenv import load_dotenv
from .vectorstore import load_vectorstore, similarity_search  # similarity_search επιστρέφει [(doc, raw_score), ...]
from .llm import load_llm, generate_answer
from rank_bm25 import BM25Okapi

# --- ΝΕΑ imports για extractive snippets ---
import regex as re2
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
# -------------------------------------------

# Φόρτωση .env (ώστε τα flags να διαβάζονται σωστά)
load_dotenv()

# Feature flags & ρυθμίσεις
RERANK_BM25_ENABLED = os.getenv("RERANK_BM25_ENABLED", "true").lower() == "true"
EXTRACTIVE_ENABLED  = os.getenv("EXTRACTIVE_ENABLED", "true").lower() == "true"
SNIPPETS_K          = int(os.getenv("SNIPPETS_K", "5"))
SNIPPETS_MIN_SIM    = float(os.getenv("SNIPPETS_MIN_SIM", "0.15"))

# Προαιρετικό (βαρύτερο): cross-encoder re-rank (δεν χρησιμοποιείται εδώ ακόμη)
CROSS_ENCODER_ENABLED = os.getenv("CROSS_ENCODER_ENABLED", "false").lower() == "true"
CROSS_ENCODER_MODEL   = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# πολύ απλή tokenization για en/el
_punct_re = _re_tok.compile(r"[^\wΆ-ώ]+", _re_tok.UNICODE)

def _tok(text: str):
    t = (text or "").lower()
    t = _punct_re.sub(" ", t)
    return [w for w in t.split() if len(w) > 1]

def _bm25_rerank(hits_with_raw, query: str, top_m: int = 10):
    """
    hits_with_raw: [(doc, raw_score), ...] από FAISS
    επιστρέφει ΙΔΙΑ δομή αλλά re-ordered από BM25 πάνω σε (title+content).
    """
    if not hits_with_raw:
        return hits_with_raw
    docs = [d for d, _ in hits_with_raw]
    corpus = [_tok(f"{d.metadata.get('title','')} {d.page_content or ''}") for d in docs]
    bm25 = BM25Okapi(corpus)
    qtok = _tok(query)
    scores = bm25.get_scores(qtok)
    ranked = sorted(zip(hits_with_raw, scores), key=lambda x: x[1], reverse=True)
    reord = [pair for (pair, _score) in ranked[:top_m]]
    return reord

# Ρύθμιση καταγραφής σφαλμάτων
logging.basicConfig(level=logging.INFO)

# 🔹 Φόρτωση vectorstore στο startup (ίδιο όπως πριν)
vectorstore = load_vectorstore()

# 🔹 Lazy-loading του LLM (ίδιο όπως πριν)
llm_instance = None
def get_llm():
    global llm_instance
    if llm_instance is None:
        llm_instance = load_llm()
    return llm_instance


# =========================
# Topic helpers
# =========================
TOPIC_KEYWORDS = {
    "ai": ["ai", "artificial intelligence", "machine learning", "neural", "open-source ai", "τεχνητή νοημοσύνη"],
    "politics": ["politics", "political", "policy", "minister", "government", "βουλή", "πολιτική", "κόμμα"],
    "sports": ["sports", "σπορ", "athletics", "match", "game", "team", "league", "πρωτάθλημα"],
    "football": ["football", "soccer", "ποδόσφ", "goal", "match", "league", "premier league", "uefa", "champions"],
    "greece": ["greece", "greek", "athens", "ελλάδα", "ελλάδα", "αθήνα", "κυβέρνηση", "ελλην"],
}

def _keywords_for_query(q: str):
    ql = (q or "").lower()
    found = []
    for _topic, kws in TOPIC_KEYWORDS.items():
        if any(k in ql for k in kws):
            found.extend(kws)
    return list(set(found))

def _text_contains_any(text: str, kws) -> bool:
    if not kws:
        return True
    tl = (text or "").lower()
    return any(k in tl for k in kws)


# (ΔΙΑΤΗΡΕΙΤΑΙ για μελλοντική χρήση)
def filter_recent_documents(documents, days=30, desired_category=None):
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    filtered_docs = []
    for doc in documents:
        metadata = doc.metadata
        published_date_str = metadata.get("published_date")
        category = metadata.get("category", "General").lower()
        if not published_date_str:
            continue
        try:
            published_date = parser.parse(published_date_str)
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=timezone.utc)
            if published_date >= cutoff_date:
                if desired_category:
                    if category == desired_category.lower():
                        filtered_docs.append((doc, published_date))
                else:
                    filtered_docs.append((doc, published_date))
        except Exception as e:
            logging.error(f"⚠️ Σφάλμα parsing ημερομηνίας: {published_date_str} -> {e}")
            continue
    logging.info(f"🧪 Νέα φίλτρα άρθρων: {len(filtered_docs)} / {len(documents)}")
    return filtered_docs


# =========================
# Scoring params & helpers
# =========================
MAX_AGE_DAYS = 45
RECENCY_WEIGHT = 0.25   # 25% επίδραση φρεσκάδας
TAU_DAYS = 15.0         # εκθετική αποσύνθεση (όσο μικρότερο, τόσο «τιμωρεί» τα παλιά)
MIN_SIM = 0.20          # κατώφλι ομοιότητας (μετά τον μετασχηματισμό)
MIN_LEN = 300           # ελάχιστο καθαρό μήκος κειμένου

def _to_utc(dt_str):
    """Μετατρέπει οποιοδήποτε date string σε aware UTC datetime."""
    if not dt_str:
        return None
    dt = parser.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _age_days(dt_str):
    dt = _to_utc(dt_str)
    if not dt:
        return 10**9
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)

def _recency_boost(dt_str):
    """0..1 boost: 1 για πολύ πρόσφατο, ~0 για παλιό. Κόβουμε άρθρα >MAX_AGE_DAYS."""
    age = _age_days(dt_str)
    if age > MAX_AGE_DAYS:
        return 0.0
    return math.exp(-age / TAU_DAYS)

def _distance_to_similarity(raw_score: float) -> float:
    """
    Μετατροπή FAISS distance -> pseudo-similarity ∈ [0,1].
    """
    try:
        d = float(raw_score)
    except Exception:
        d = 1.0
    return 1.0 / (1.0 + d) if d > 1.0 else max(0.0, 1.0 - d)

def _final_score(similarity: float, published_iso: str) -> float:
    r = _recency_boost(published_iso)
    return similarity * (1.0 - RECENCY_WEIGHT) + RECENCY_WEIGHT * r


def _select_single_article(hits_with_raw, query_kws=None):
    """
    hits_with_raw: λίστα από tuples (doc, raw_score)
    -> επιστρέφει (doc, topic_match: bool) ή (None, False)
    """
    query_kws = query_kws or []
    candidates = []
    for doc, raw in hits_with_raw:
        sim = _distance_to_similarity(raw)
        if sim < MIN_SIM:
            continue
        text = (doc.page_content or "").strip()
        if len(text) < MIN_LEN:
            continue
        pub = doc.metadata.get("published_date")
        if _recency_boost(pub) == 0.0:
            continue

        # topic match: τουλάχιστον ένα keyword μέσα στο title+content
        topic_match = _text_contains_any(f"{doc.metadata.get('title','')} {text}", query_kws)

        score = _final_score(sim, pub) + (0.03 if topic_match and query_kws else 0.0)  # μικρό bonus αν ταιριάζει
        candidates.append((score, doc, sim, pub, topic_match))

    if not candidates:
        return None, False

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_doc, best_sim, best_pub, best_match = candidates[0]
    logging.info(
        f"🔎 Επιλέχθηκε άρθρο: '{best_doc.metadata.get('title','—')}' "
        f"| sim={best_sim:.3f}, rec={_recency_boost(best_pub):.3f}, final={best_score:.3f}, "
        f"date={best_pub}, topic_match={best_match}"
    )
    return best_doc, best_match


# =========================
# Extractive snippets helpers (ΝΕΑ)
# =========================
_embedder = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

# απλός splitter που δουλεύει en/el (., ?, !, ελληνική άνω τελεία '·', ellipsis …)
_SENT_SPLIT_RE = re2.compile(r'(?<=[\.\?\!·…])\s+(?=[A-ZΑ-ΩΉΊΌΎΏΆΈΉΊΌΎΏ])')

def _split_sentences(text: str):
    txt = (text or "").strip()
    parts = _SENT_SPLIT_RE.split(txt) if txt else []
    # πέτα πολύ κοντές/σκουπιδένιες προτάσεις και κόψε υπερβολικό μήκος
    return [s.strip() for s in parts if len(s.strip()) >= 40][:60]

def _cos(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def _mmr_select(sentences, query_vec, sent_vecs, k=4, lambda_param=0.7):
    if not sentences:
        return [], []
    sims = np.array([_cos(v, query_vec) for v in sent_vecs])
    selected = []
    candidates = list(range(len(sentences)))
    # πρώτο: max relevance
    first = int(np.argmax(sims))
    selected.append(first)
    candidates.remove(first)
    while len(selected) < min(k, len(sentences)):
        mmr_scores = []
        for i in candidates:
            rel = sims[i]
            div = max(_cos(sent_vecs[i], sent_vecs[j]) for j in selected)
            mmr = lambda_param * rel - (1.0 - lambda_param) * div
            mmr_scores.append((mmr, i))
        mmr_scores.sort(reverse=True)
        best_i = mmr_scores[0][1]
        selected.append(best_i)
        candidates.remove(best_i)
    return [sentences[i] for i in selected], [sims[i] for i in selected]

def _extractive_snippets(doc, user_query, k=4, min_sim=0.20):
    text = (doc.page_content or "").strip()
    sents = _split_sentences(text)
    if not sents:
        return []
    emb = _get_embedder()
    try:
        sent_vecs = np.array(emb.embed_documents(sents))
        query_vec = np.array(emb.embed_query(user_query))
    except Exception as _e:
        logging.warning(f"Embed error (snippets): {_e}")
        return []
    chosen, sims = _mmr_select(sents, query_vec, sent_vecs, k=k, lambda_param=0.7)
    if chosen and float(np.mean(sims)) >= min_sim:
        return chosen
    return []


def _build_prompt_one_article(user_query: str, doc, lang: str, topic_match: bool) -> str:
    title = doc.metadata.get("title", "—")
    link = doc.metadata.get("link", "")
    pub_iso_dt = _to_utc(doc.metadata.get("published_date"))
    pub_iso = pub_iso_dt.isoformat() if pub_iso_dt else "Άγνωστη ημερομηνία"
    content = (doc.page_content or "").strip()[:1200]

    if lang == "en":
        return f"""
[SYSTEM]
You are a news assistant. Answer STRICTLY from the SINGLE article text below.

OUTPUT RULES (MUST FOLLOW EXACTLY):
- Write a single paragraph with 3–5 sentences. No headings, no lists, no labels.
- First sentence MUST begin with: "Based on the {pub_iso} article:"
- Use ONLY the article. Do NOT invent facts. Do NOT ask the user questions.
- Mention the UTC date only in the first sentence. Include the source link only ONCE as the LAST token: {link}
- Do NOT mention internal flags or sections. Do NOT write the words "Topic match", "Title", "Source", "Excerpt", or similar.
- If and only if the article does NOT cover the user's topic directly, add one short note as the LAST sentence before the link: "Note: the provided article does not cover the requested topic directly."
- Do NOT produce lists or bullet points under any circumstance.

[ARTICLE TEXT]
Title: {title}
Date (UTC): {pub_iso}
Link: {link}
Excerpt:
\"\"\"
{content}
\"\"\"

[USER QUESTION]
{user_query}

[INTERNAL FLAG — DO NOT MENTION IN THE ANSWER]
topic_match={str(topic_match).lower()}
""".strip()

    # default: Greek
    return f"""
[ΣΥΣΤΗΜΑ]
Είσαι βοηθός ειδήσεων. Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ από το παρακάτω ΕΝΑ άρθρο.

ΚΑΝΟΝΕΣ ΕΞΟΔΟΥ (ΑΚΡΙΒΩΣ ΕΤΣΙ):
- Γράψε ένα ενιαίο παράγραφο με 3–5 προτάσεις. Χωρίς επικεφαλίδες, χωρίς λίστες, χωρίς labels.
- Η πρώτη πρόταση ΠΡΕΠΕΙ να ξεκινά με: "Βάσει άρθρου της {pub_iso}:"
- Χρησιμοποίησε ΜΟΝΟ το άρθρο. ΜΗΝ εφευρίσκεις στοιχεία. ΜΗΝ ρωτάς τον χρήστη.
- Ανάφερε την ημερομηνία UTC μόνο στην πρώτη πρόταση. Βάλε το link ΜΟΝΟ ΜΙΑ φορά ως ΤΕΛΕΥΤΑΙΟ token: {link}
- ΜΗΝ αναφέρεις εσωτερικές σημαίες/sections. ΜΗΝ γράψεις λέξεις όπως "Σημαία", "Topic match", "Τίτλος", "Πηγή", "Απόσπασμα".
- ΜΟΝΟ αν το άρθρο δεν ταιριάζει άμεσα το ζητούμενο θέμα, πρόσθεσε ΜΙΑ σύντομη τελική πρόταση πριν το link: "Σημείωση: το άρθρο δεν καλύπτει άμεσα το ζητούμενο θέμα."
- ΜΗΝ χρησιμοποιείς λίστες ή bullets για κανέναν λόγο.

[ΚΕΙΜΕΝΟ ΑΡΘΡΟΥ]
Τίτλος: {title}
Ημερομηνία (UTC): {pub_iso}
Σύνδεσμος: {link}
Απόσπασμα:
\"\"\"
{content}
\"\"\"

[ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ]
{user_query}

[ΕΣΩΤΕΡΙΚΗ ΣΗΜΑΙΑ — ΜΗΝ ΤΗΝ ΑΝΑΦΕΡΕΙΣ ΣΤΗΝ ΑΠΑΝΤΗΣΗ]
topic_match={str(topic_match).lower()}
""".strip()


# --- ΝΕΟ prompt για extractive snippets ---
def _build_prompt_from_snippets(user_query: str, doc, snippets: list[str], lang: str) -> str:
    title = doc.metadata.get("title", "—")
    link = doc.metadata.get("link", "")
    pub_iso_dt = _to_utc(doc.metadata.get("published_date"))
    pub_iso = pub_iso_dt.isoformat() if pub_iso_dt else "Άγνωστη ημερομηνία"
    joined = " ".join(snippets)[:1200]

    if lang == "en":
        return f"""
[SYSTEM]
Rewrite the provided sentences into a concise, neutral-tone news answer. No emojis, no exclamation tone.

RULES:
- ONE paragraph, 3–5 sentences.
- First sentence MUST begin: "Based on the {pub_iso} article:"
- Use ONLY the sentences below (no outside facts).
- Mention the date only in the first sentence.
- End with the source link as the LAST token: {link}
- Do NOT add disclaimers unless the sentences are irrelevant to the question.
- Do NOT produce lists or bullet points under any circumstance.

[SENTENCES]
\"\"\"
{joined}
\"\"\"

[USER QUESTION]
{user_query}
""".strip()

    return f"""
[ΣΥΣΤΗΜΑ]
Ξαναγράψε τις παρεχόμενες προτάσεις σε συνοπτική, ουδέτερου ύφους ειδησεογραφική απάντηση. Χωρίς emojis.

ΚΑΝΟΝΕΣ:
- Ένας παράγραφος, 3–5 προτάσεις.
- Η πρώτη ΠΡΕΠΕΙ να αρχίζει: "Βάσει άρθρου της {pub_iso}:"
- Χρησιμοποίησε ΜΟΝΟ τις προτάσεις (καμία εξωτερική πληροφορία).
- Ανάφερε την ημερομηνία μόνο στην πρώτη πρόταση.
- Κλείσε με το link ως ΤΕΛΕΥΤΑΙΟ token: {link}
- ΜΗΝ προσθέτεις disclaimers εκτός αν οι προτάσεις είναι άσχετες με την ερώτηση.
- ΜΗΝ χρησιμοποιείς λίστες ή bullets για κανέναν λόγο.

[ΠΡΟΤΑΣΕΙΣ]
\"\"\"
{joined}
\"\"\"

[ΕΡΩΤΗΣΗ]
{user_query}
""".strip()


# 🔹 Βασική RAG λειτουργία (ΔΙΑΤΗΡΕΙΤΑΙ ΟΝΟΜΑ & ΥΠΟΓΡΑΦΗ)
def generate_contextual_answer(user_query, category: str = None):
    import re
    user_query = str(user_query).strip()

    # Ελάχιστος έλεγχος ποιότητας ερώτησης
    if not user_query or len(user_query) < 3:
        return "Η ερώτησή σου είναι πολύ μικρή ή άδεια. Δοκίμασε να ρωτήσεις κάτι πιο συγκεκριμένο."
    if re.fullmatch(r'[\W_]+', user_query):
        return "Η ερώτησή σου περιέχει μόνο σύμβολα. Δοκίμασε ξανά με λέξεις."
    if user_query.lower() in {"hello", "hi", "hey", "γεια", "καλησπέρα"}:
        return "Γεια σου! Μπορείς να με ρωτήσεις κάτι για τις ειδήσεις, όπως \"Τι νέα για την τεχνητή νοημοσύνη;\""

    try:
        if vectorstore is None:
            return "Σφάλμα: Το vectorstore δεν είναι διαθέσιμο."

        # 1) Similarity search ΜΕ score (παίρνουμε tuples) — αυξημένο k για καλύτερο recall
        t0 = perf_counter()
        hits_with_raw = similarity_search(user_query, k=40)
        t_faiss = (perf_counter() - t0) * 1000.0

        if not hits_with_raw:
            return _fallback_no_context(user_query)

        # 1.5) Keyword boost (ήπιο) βάσει του query
        query_kws = _keywords_for_query(user_query)
        if query_kws:
            boosted = []
            for doc, raw in hits_with_raw:
                text = f"{doc.metadata.get('title','')} {doc.page_content or ''}".lower()
                overlap = sum(1 for k in query_kws if k in text)
                new_raw = max(0.0, float(raw) - 0.03 * overlap)  # 0.03 ανά keyword
                boosted.append((doc, new_raw))
            hits_with_raw = boosted

        # 2) Category boost (ήπιο) βάσει category param
        if category:
            cat = category.lower().strip()
            boosted = []
            for doc, raw in hits_with_raw:
                doc_cat = (doc.metadata.get("category") or "").lower()
                bonus = 0.05 if doc_cat == cat else 0.0
                new_raw = max(0.0, float(raw) - bonus)
                boosted.append((doc, new_raw))
            hits_with_raw = boosted

        # 2.5) BM25 re-rank (μόνο αν είναι ενεργό το flag)
        if RERANK_BM25_ENABLED:
            t1 = perf_counter()
            hits_with_raw = _bm25_rerank(hits_with_raw, user_query, top_m=10)
            t_bm25 = (perf_counter() - t1) * 1000.0
        else:
            t_bm25 = 0.0

        # 3) Επιλογή ΕΝΟΣ άρθρου με thresholds + recency + topic flag
        chosen, topic_match = _select_single_article(hits_with_raw, query_kws=query_kws)
        if not chosen:
            return _fallback_no_context(user_query)

        # 4) Prompt για ΕΝΑ άρθρο — ΠΡΩΤΑ προσπαθούμε extractive snippets
        lang = _detect_lang(user_query)
        snippets = _extractive_snippets(chosen, user_query, k=SNIPPETS_K, min_sim=SNIPPETS_MIN_SIM) if EXTRACTIVE_ENABLED else []
        if snippets:
            prompt = _build_prompt_from_snippets(user_query, chosen, snippets, lang)
        else:
            prompt = _build_prompt_one_article(user_query, chosen, lang, topic_match)

        # 5) Γεννήτρια LLM
        t2 = perf_counter()
        answer = generate_answer(get_llm(), prompt)
        t_llm = (perf_counter() - t2) * 1000.0

        # ασφάλεια τύπου
        if not isinstance(answer, str):
            answer = str(answer or "")
        answer = answer.strip()

        # fallback αν άδειο/πολύ μικρό
        if not answer or len(answer) < 10:
            return _fallback_no_context(user_query)

        # 6) Post-processing: αφαίρεση τυχόν “Topic match:” / labels
        answer = _re.sub(r'(?im)^\s*(topic\s*match\s*:.*)$', '', answer).strip()
        answer = _re.sub(
            r'(?im)^\s*(title|date\s*\(utc\)|link|source|excerpt|άρθρο|τίτλος|ημερομηνία|σύνδεσμος|απόσπασμα)\s*:\s*.*$',
            '',
            answer
        ).strip()
        # συμπτύξεις πολλαπλών κενών γραμμών
        answer = _re.sub(r'\n{2,}', '\n', answer).strip()

        logging.info(f"[timings] faiss_ms={t_faiss:.1f} bm25_ms={t_bm25:.1f} llm_ms={t_llm:.1f}")
        return answer

    except Exception as e:
        logging.exception(f"Σφάλμα κατά την επεξεργασία της ερώτησης: {str(e)}")
        return f"Σφάλμα κατά την επεξεργασία της ερώτησης: {str(e)}"


def _detect_lang(q: str) -> str:
    # απλή ευρετική: αν έχει ελληνικούς χαρακτήρες -> el, αλλιώς en
    return "el" if re.search(r"[Α-Ωα-ω]", q or "") else "en"

def _fallback_no_context(q: str) -> str:
    return ("No sufficiently recent/relevant article (within 30 days). "
            "Try a more specific query or another topic.") if _detect_lang(q) == "en" else (
            "Δεν βρέθηκε αρκετά πρόσφατο/σχετικό άρθρο (εντός 30 ημερών). "
            "Δοκίμασε πιο συγκεκριμένη αναζήτηση ή άλλο θέμα.")
