# backend/app/api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Tuple
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
import logging
import re
import random
from ..utils_text import get_source_from_link
from urllib.parse import urlparse
from ..chatbot.vectorstore import reload_vectorstore, get_vectorstore_info 
from ..chatbot.rag import generate_contextual_answer  
from ..chatbot.vectorstore import load_vectorstore
from ..chatbot.llm import load_llm, generate_answer
router = APIRouter()
logger = logging.getLogger(__name__)

# Public root

@router.get("/")
async def read_root():
    return {"message": "Talk2News Chatbot API is up and running!"}

# Ask 

class Question(BaseModel):
    question: str
    category: str | None = None

@router.post("/ask")
async def ask_question(question: Question):
    """
    Default: τρέχει το υπάρχον RAG για ένα άρθρο.
    (Κρατάμε την παλιά συμπεριφορά ώστε να μην "σπάσει" τίποτα.)
    """
    answer = generate_contextual_answer(question.question, question.category)
    return {"answer": answer}

# Weekly Sum 
WORLD_NEGATIVE_HINTS = [
    "ελλάδα", "αθήνα", "ελλην", "κυβέρνηση της ελλάδας", "παναθηναϊκ", "ολυμπιακ", "αεκ",
    "greece", "greek", "athens"
]

def _is_world_article(meta: dict, text: str) -> bool:
    title = (meta.get("title") or "").lower()
    combined = f"{title} {(text or '').lower()}"
    return not any(tok in combined for tok in WORLD_NEGATIVE_HINTS)

def _to_utc(dt_str: str | None):
    if not dt_str:
        return None
    try:
        dt = dtparser.parse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _domain_of(link: str) -> str:
    try:
        return urlparse(link).netloc.lower().replace("www.", "")
    except Exception:
        return ""

def _norm_title(t: str) -> str:
    t = (t or "").strip().lower()
    # απλό normalization/decoration strip
    t = re.sub(r"\s+", " ", t)
    return t

def _multiquery_pool(vs, window_hours: int) -> List[Tuple[datetime, dict, str]]:
    """
    Ρίχνουμε πολλά queries για ποικιλία, φιλτράρουμε στο παράθυρο, world-only,
    και επιστρέφουμε [(published_utc, meta, text), ...]
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    queries = [
        # γεωπολιτική/πολιτική/κόσμος
        "world news international headlines geopolitics diplomacy elections last 7 days",
        # οικονομία/αγορές
        "global economy business markets inflation central banks last 7 days",
        # τεχνολογία/AI
        "technology artificial intelligence cybersecurity big tech last 7 days",
        # επιστήμη/κλίμα/υγεία
        "science climate environment health medicine last 7 days",
        # συγκρούσεις/πόλεμοι
        "conflicts wars middle east ukraine israel gaza last 7 days",
        # αθλητισμός παγκοσμίως
        "sports world tournaments results football basketball tennis last 7 days",
    ]

    pool: List[Tuple[datetime, dict, str]] = []
    for q in queries:
        try:
            # Παίρνουμε αρκετά για να υπάρχει υλικό για diversity/dedup.
            hits = vs.similarity_search_with_score(q, k=100)
        except Exception as e:
            logger.warning(f"FAISS search error for '{q}': {e}")
            continue

        for doc, _score in hits:
            meta = doc.metadata or {}
            pub = _to_utc(meta.get("published_date"))
            if not pub or pub < cutoff:
                continue
            txt = (doc.page_content or "").strip()
            if not txt:
                continue
            #if not _is_world_article(meta, txt):
                #continue
            pool.append((pub, meta, txt))

    return pool

def _dedup_and_diversify(pool: List[Tuple[datetime, dict, str]],
                         max_items: int = 6,
                         per_domain_limit: int = 2) -> List[Tuple[dict, str]]:
    """
    - Sort by recency
    - Dedup by (link or normalized title)
    - Diversity: περιορίζουμε max N ανά domain
    """
    if not pool:
        return []

    # νεότερα πρώτα
    pool.sort(key=lambda x: x[0], reverse=True)

    # dedup
    seen_keys = set()
    deduped: List[Tuple[datetime, dict, str]] = []
    for pub, meta, txt in pool:
        link = meta.get("link") or ""
        ttl = _norm_title(meta.get("title") or "")
        key = link or ttl
        if not key:
            # αν δεν έχει τίτλο/λινκ, χρησιμοποίησε timestamp + hash text
            key = f"{pub.isoformat()}::{hash(txt[:120])}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((pub, meta, txt))

    if not deduped:
        return []

    # diversity per domain
    per_domain_count = {}
    picked: List[Tuple[dict, str]] = []
    for _pub, meta, txt in deduped:
        dom = _domain_of(meta.get("link") or "")
        c = per_domain_count.get(dom, 0)
        if c >= per_domain_limit:
            continue
        per_domain_count[dom] = c + 1
        picked.append((meta, txt))
        if len(picked) >= max_items:
            break

    return picked

def _build_weekly_digest_prompt(items: List[Tuple[dict, str]], lang: str = "en") -> str:
    """
     prompt για daily digest - εστίαση σε σημαντικά νέα
    """
    def _brief(txt: str) -> str:
        t = (txt or "").replace("\n", " ").strip()
        return (t[:200] + "…") if len(t) > 200 else t

    lines = []
    for meta, txt in items:
        title = (meta.get("title") or "—").replace("\n", " ").strip()
        link = meta.get("link", "")
        pub = _to_utc(meta.get("published_date"))
        pub_iso = pub.isoformat() if pub else ""
        source = get_source_from_link(link)
        
        lines.append(f"- {title} ({pub_iso}) — {_brief(txt)} [source: {source}]")

    joined = "\n".join(lines)
    n = len(items)

    if lang == "el":
        return f"""
[ΣΥΣΤΗΜΑ]
Δημιούργησε μια σύντομη ενημερωτική σύνοψη των κορυφαίων ειδήσεων των τελευταίων 48 ωρών.

ΚΑΝΟΝΕΣ:
- Επίλεξε τα 5-6 πιο ΣΗΜΑΝΤΙΚΑ νέα (πολιτικά, οικονομικά, διεθνή, τεχνολογία)
- Αγνόησε μικρές ή ασήμαντες ειδήσεις
- Δώσε προτεραιότητα σε πρόσφατα γεγονότα με ευρεία επίδραση
- Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες από τα παρεχόμενα άρθρα
- Γράψε ΕΝΑν συνεκτικό παράγραφο 4-6 προτάσεων

[ΕΙΔΗΣΕΙΣ ({n})]
{joined}

Σύνοψη κορυφαίων ειδήσεων:
""".strip()

    return f"""
[SYSTEM]
Create a concise daily digest of the top news items from the last 48 hours.

RULES:
- Select the 5-6 most IMPORTANT news (politics, economy, international, technology)  
- Ignore minor or trivial news stories
- Prioritize recent events with broad impact
- Use ONLY information from the provided articles
- Write ONE coherent paragraph of 4-6 sentences

[NEWS ITEMS ({n})]
{joined}

Daily news digest:
""".strip()

def build_weekly_digest_answer(max_items: int = 6, hours: int = 7 * 24, lang: str = "en") -> str:
    """
    Πλήρης ροή: multi-query pool -> dedup/diversity -> LLM synthesis (1 paragraph).
    """
    vs = load_vectorstore()
    if not vs:
        return "Vector index is not available right now."

    #  Πάρε μεγάλο pool από πολλές κατηγορίες
    pool = _multiquery_pool(vs, window_hours=hours)
    if not pool:
        return "No sufficiently recent world news found in the selected window."

    # Time-seeded shuffle για ποικιλία (αλλά σταθερότητα μέσα στην ημέρα)
    # Random shuffle για ποικιλία σε κάθε κλήση
    random.shuffle(pool)

    #  Dedup + diversity
    items = _dedup_and_diversify(pool, max_items=max_items, per_domain_limit=2)
    if not items:
        return "No sufficiently diverse world news found."

    # Prompt & LLM
    prompt = _build_weekly_digest_prompt(items, lang=lang)
    llm = load_llm()
    answer = generate_answer(llm, prompt).strip()
    return answer or "No digest available right now."


@router.get("/digest")
def weekly_digest(n: int = 6, hours: int = 48, lang: str = "en"):
    """
    Επιστρέφει daily digest των τελευταίων ωρών
    """
    n = max(5, min(n, 7))
    lang = "el" if re.search(r"[Α-Ωα-ω]", lang or "") else "en"
    
    print(f" DIGEST ENDPOINT CALLED: n={n}, hours={hours}, lang={lang}")
    
    digest = build_weekly_digest_answer(max_items=n, hours=hours, lang=lang)
    
    print(f" DIGEST GENERATED: {len(digest)} characters")
    
    return {"digest": digest}

@router.post("/reload-vectorstore")
async def reload_vectorstore_endpoint():
    """
    Manual reload του vectorstore cache.
    """
    try:
        vs = reload_vectorstore()
        if vs:
            doc_count = len(vs.docstore._dict)
            return {
                "status": "success", 
                "message": f"Vectorstore reloaded successfully! ({doc_count} documents)"
            }
        else:
            return {"status": "error", "message": "Vectorstore reload failed"}
    except Exception as e:
        logger.error(f"Vectorstore reload error: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/vectorstore-status")
async def vectorstore_status_endpoint():
    """
    Επιστρέφει την τρέχουσα κατάσταση του vectorstore.
    """
    return get_vectorstore_info()