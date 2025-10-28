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
from .vectorstore import load_vectorstore, similarity_search  
from .llm import load_llm, generate_answer
from rank_bm25 import BM25Okapi
import regex as re2
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Φόρτωση .env (ώστε τα flags να διαβάζονται σωστά)
load_dotenv()

# Feature flags & ρυθμίσεις
RERANK_BM25_ENABLED = os.getenv("RERANK_BM25_ENABLED", "true").lower() == "true"
EXTRACTIVE_ENABLED  = os.getenv("EXTRACTIVE_ENABLED", "true").lower() == "true"
SNIPPETS_K          = int(os.getenv("SNIPPETS_K", "5"))
SNIPPETS_MIN_SIM    = float(os.getenv("SNIPPETS_MIN_SIM", "0.15"))

# cross-encoder re-rank 
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

# 🔹 Φόρτωση vectorstore στο startup
vectorstore = load_vectorstore()

# 🔹 Lazy-loading του LLM
llm_instance = None
def get_llm():
    global llm_instance
    if llm_instance is None:
        llm_instance = load_llm()
    return llm_instance

# Topic helpers

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

# Scoring params & helpers 
MAX_AGE_DAYS = 30        
RECENCY_WEIGHT = 0.5 
TAU_DAYS = 3.0         
MIN_SIM = 18.0         
MIN_LEN = 300           

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
    
    age = _age_days(dt_str)
   
    if age > 40:  
        return 0.0
   
    if age <= 1:    # Σήμερα
        return 0.8  # Από 1.0
    elif age <= 3:  # 1-3 ημέρες
        return 0.6  # Από 0.9
    elif age <= 7:  # 1 εβδομάδα
        return 0.4  # Από 0.7
    elif age <= 14: # 2 εβδομάδες
        return 0.2  # Από 0.4
    elif age <= 30: # 1 μήνας
        return 0.1  # Νέο
    else:           # 1-2 μήνες
        return 0.05 # Νέο

def _distance_to_similarity(raw_score: float) -> float:
    """
    Μετατροπή FAISS distance -> pseudo-similarity ∈ [0,1].
    """
    try:
        d = float(raw_score)
    except Exception:
        d = 1.0
    return 1.0 / (1.0 + d) if d > 1.0 else max(0.0, 1.0 - d)

def _final_score(similarity: float, published_iso: str, query: str = "") -> float:
    
    r = _recency_boost(published_iso)
    
    # DYNAMIC WEIGHTING based on query
    query_lower = (query or "").lower()
    
    # Αν η ερώτηση ζητάει ρητά πρόσφατα νέα
    if any(keyword in query_lower for keyword in 
           ["today", "latest", "recent", "breaking", "just", "new", "πρόσφατα", "σήμερα", "νέα"]):
        recency_weight = 0.65  #  Υψηλότερη προτεραιότητα σε recency
    else:
        recency_weight = RECENCY_WEIGHT  # Normal weight
    
    return similarity * (1.0 - recency_weight) + recency_weight * r


def _select_single_article(hits_with_raw, query_kws=None, query: str = "", *, hard_recency=False, min_sim=0.10):
    """
    ΕΝΗΜΕΡΩΜΕΝΗ: Πιο ελεύθερη επιλογή άρθρων με έμφαση στο relevance παρά στο recency
    """
    query_kws = query_kws or []
    candidates = []
    
    current_date = datetime.now(timezone.utc)
    
    print(f"🔍 FILTERING ARTICLES - Current date: {current_date.date()}")
    print(f"📊 Processing {len(hits_with_raw)} hits | hard_recency: {hard_recency} | min_sim: {min_sim}")

    for i, (doc, raw) in enumerate(hits_with_raw):
        sim = _distance_to_similarity(raw)
        
        # Βασικό similarity filtering
        if sim < min_sim:
            continue

        text = (doc.page_content or "").strip()
        if len(text) < 200:  # Ελαφρώς μικρότερο minimum
            continue

        pub = doc.metadata.get("published_date")
        pub_dt = _to_utc(pub)
        
        if not pub_dt:
            continue
            
        # Recency calculation
        age_days = (current_date - pub_dt).days
        
        # ΜΟΝΟ αν hard_recency=True, απορρίπτουμε παλιά άρθρα
        if hard_recency and age_days > 14:
            continue

        # Πολύ πιο ελεύθερο title filtering
        title = doc.metadata.get("title", "").lower()
        generic_indicators = ["home", "page", "archive"]  # Λιγότεροι δείκτες
        if any(indicator in title for indicator in generic_indicators) and len(title) < 15:
            continue

        # Topic matching
        topic_match = _text_contains_any(f"{doc.metadata.get('title','')} {text}", query_kws)
    
        # ΒΕΛΤΙΩΜΕΝΟ SCORING:
        base_score = _final_score(sim, pub, query)
        
        # Μέτριο bonus για topic match
        if topic_match and query_kws:
            base_score += 0.08
            
        # Μέτριο bonus για πρόσφατα άρθρα
        if age_days <= 7:
            base_score += 0.10
        elif age_days <= 14:
            base_score += 0.05
            
        # Πολύ μικρό penalty για παλιά άρθρα
        if age_days > 30:
            base_score *= 0.9
            
        # Bonus για query-specific relevance
        if "ai" in query.lower() and any(ai_term in text.lower() for ai_term in ["ai", "artificial", "openai", "chatgpt"]):
            base_score += 0.15
            
        if "trump" in query.lower() and any(pol_term in text.lower() for pol_term in ["trump", "president", "election", "white house"]):
            base_score += 0.15
            
        candidates.append((base_score, doc, sim, pub, topic_match, age_days))

    if not candidates:
        print("No candidates passed basic filtering - using fallback")
        # Fallback: πάρε το κορυφαίο από similarity search
        if hits_with_raw:
            best_doc = hits_with_raw[0][0]
            pub_date = best_doc.metadata.get("published_date", "No date")
            title = best_doc.metadata.get('title', 'No title')[:60]
            print(f"🔄 FALLBACK: Using top similarity result: {pub_date} | '{title}...'")
            return best_doc, False
        return None, False

    # Ταξινόμηση και επιλογή
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    print("TOP CANDIDATES:")
    for i, (score, doc, sim, pub, topic_match, age) in enumerate(candidates[:5]):
        title = doc.metadata.get('title', 'No title')
        source = doc.metadata.get('source', 'Unknown')
        print(f"  {i+1}. Score: {score:.3f} | Age: {age}d | Sim: {sim:.3f} | {source} | '{title[:50]}...'")

    best_score, best_doc, best_sim, best_pub, best_match, best_age = candidates[0]
    
    print(f"SELECTED: '{best_doc.metadata.get('title','—')}' | Score: {best_score:.3f} | Age: {best_age}d")
    return best_doc, best_match


# Extractive snippets helpers 

_embedder = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        # ⚠️ ΣΥΜΦΩΝΙΑ ΜΕ ΤΟ FAISS: χρησιμοποιούμε το ίδιο embedding model (all-mpnet-base-v2)
        _embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
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

def enhance_query_for_search(query: str) -> str:
    """
    Βελτιώνει τα queries για καλύτερη αναζήτηση
    """
    query_lower = query.lower()
    
    # 🔥 ΑΛΛΑΓΗ: Πιο συγκεκριμένα keywords
    if any(word in query_lower for word in ["latest", "recent", "new", "today", "breaking"]):
        return query + " news updates 2025 current"
    elif any(word in query_lower for word in ["trump", "biden", "politics"]):
        return query + " election president US America 2025"
    
    # Γενική βελτίωση
    return query


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


# 🔹 Βασική RAG λειτουργία 
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

        # 🔍 CRITICAL DEBUG: Check what articles we're finding
        print(f"🔍 RAG QUERY: '{user_query}'")
        
        # Similarity search με score 
        t0 = perf_counter()
        hits_with_raw = similarity_search(user_query, k=40)
        t_faiss = (perf_counter() - t0) * 1000.0

        #  DEBUG: Show top search results with dates
        print(" TOP 10 SEARCH RESULTS:")
        for i, (doc, score) in enumerate(hits_with_raw[:10]):
            title = doc.metadata.get('title', 'No title')[:70]
            date = doc.metadata.get('published_date', 'No date')
            source = doc.metadata.get('link', 'No link')[:40]
            
            # Calculate actual age
            try:
                pub_dt = _to_utc(date)
                if pub_dt:
                    now = datetime.now(timezone.utc)
                    age_days = (now - pub_dt).days
                else:
                    age_days = "N/A"
            except:
                age_days = "N/A"
                
            print(f"  {i+1}. '{title}'...")
            print(f"     Date: {date} | Age: {age_days} days | Source: {source}")
            print(f"     Score: {score:.3f}")
            print()

        if not hits_with_raw:
            return _fallback_no_context(user_query)

        # Keyword boost βάσει του query
        query_kws = _keywords_for_query(user_query)
        if query_kws:
            boosted = []
            for doc, raw in hits_with_raw:
                text = f"{doc.metadata.get('title','')} {doc.page_content or ''}".lower()
                overlap = sum(1 for k in query_kws if k in text)
                new_raw = max(0.0, float(raw) - 0.03 * overlap)  # 0.03 ανά keyword
                boosted.append((doc, new_raw))
            hits_with_raw = boosted

        # Category boost βάσει category param
        if category:
            cat = category.lower().strip()
            boosted = []
            for doc, raw in hits_with_raw:
                doc_cat = (doc.metadata.get("category") or "").lower()
                bonus = 0.05 if doc_cat == cat else 0.0
                new_raw = max(0.0, float(raw) - bonus)
                boosted.append((doc, new_raw))
            hits_with_raw = boosted

        # BM25 re-rank (μόνο αν είναι ενεργό το flag)
        if RERANK_BM25_ENABLED:
            t1 = perf_counter()
            hits_with_raw = _bm25_rerank(hits_with_raw, user_query, top_m=10)
            t_bm25 = (perf_counter() - t1) * 1000.0
        else:
            t_bm25 = 0.0

        # Επιλογή ενός άρθρου:
        # strict pass: σεβόμαστε recency window και κατώφλι MIN_SIM
        chosen, topic_match = _select_single_article(
            hits_with_raw,
            query_kws=query_kws,
            query=user_query,  # Pass query for dynamic weighting
            hard_recency=False,
            min_sim=0.10
        )

        # relaxed pass: αν δεν βρεθεί τίποτα, επέτρεψε παλιά άρθρα & χαμήλωσε λίγο το min_sim
        if not chosen:
            chosen, topic_match = _select_single_article(
                hits_with_raw,
                query_kws=query_kws,
                query=user_query,     # Pass query for dynamic weighting
                hard_recency=False,   # μην κόβεις παλιά
                min_sim=0.10          # δέξου ελαφρώς χαμηλότερη ομοιότητα
            )

        # last resort: αν ακόμα δεν υπάρχει, πάρε το πρώτο hit «ως έχει»
        if not chosen and hits_with_raw:
            doc0, _raw0 = hits_with_raw[0]
            chosen, topic_match = doc0, False

        if not chosen:
            return _fallback_no_context(user_query)

        # 🔍 DEBUG: Show final selected article
        if chosen:
            selected_date = chosen.metadata.get('published_date', 'No date')
            selected_title = chosen.metadata.get('title', 'No title')
            try:
                pub_dt = _to_utc(selected_date)
                if pub_dt:
                    now = datetime.now(timezone.utc)
                    age_days = (now - pub_dt).days
                else:
                    age_days = "N/A"
            except:
                age_days = "N/A"
                
            print(f"🎯 FINAL SELECTED ARTICLE:")
            print(f"   Title: {selected_title}")
            print(f"   Date: {selected_date} (Age: {age_days} days)")
            print(f"   Source: {chosen.metadata.get('link', 'No link')}")
            print()

        # Prompt για ένα άρθρο — πρώτα προσπαθούμε extractive snippets
        lang = _detect_lang(user_query)
        snippets = _extractive_snippets(chosen, user_query, k=SNIPPETS_K, min_sim=SNIPPETS_MIN_SIM) if EXTRACTIVE_ENABLED else []
        if snippets:
            prompt = _build_prompt_from_snippets(user_query, chosen, snippets, lang)
        else:
            prompt = _build_prompt_one_article(user_query, chosen, lang, topic_match)

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

        # Post-processing: αφαίρεση τυχόν labels
        answer = _re.sub(r'(?im)^\s*(topic\s*match\s*:.*)$', '', answer).strip()
        answer = _re.sub(
            r'(?im)^\s*(title|date\s*\(utc\)|link|source|excerpt|άρθρο|τίτλος|ημερομηνία|σύνδεσμος|απόσπασμα)\s*:\s*.*$',
            '',
            answer
        ).strip()
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
