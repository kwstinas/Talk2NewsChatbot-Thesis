import feedparser
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List
from pymongo import MongoClient, ASCENDING, errors
from pymongo.errors import OperationFailure
from ..utils_text import clean_html, normalize_published_date
# --------- Logging setup ---------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- RSS feeds ---------
FEEDS: Dict[str, str] = {
    "Newsbeast": "https://www.newsbeast.gr/feed",
    "Naftemporiki": "https://www.naftemporiki.gr/feed/rss",
    "Greek Reporter": "https://greekreporter.com/feed/",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "TechCrunch": "http://feeds.feedburner.com/TechCrunch/",
    "SKAI": "https://www.skai.gr/rss.xml",
    "In.gr": "https://www.in.gr/feed/",
    "ABC News World": "https://abcnews.go.com/abcnews/internationalheadlines",
    "To Vima": "https://www.tovima.gr/feed/",
    "Documento": "https://documentonews.gr/feed/",
    "Eleftheros Typos": "https://eleftherostypos.gr/feed/",
    "TaNea": "https://www.tanea.gr/feed/",
    "Associated Press": "https://feedx.net/rss/ap.xml",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "TechRadar": "https://www.techradar.com/rss",
}

# --------- Θεματικές Κατηγορίες (απλή keyword προσέγγιση) ---------
CATEGORIES: Dict[str, Iterable[str]] = {
    "Politics": ["government", "election", "minister", "policy", "βουλή", "πολιτική"],
    "Economy": ["inflation", "market", "stock", "επενδύσεις", "οικονομία", "τράπεζα"],
    "Technology": ["ai", "artificial intelligence", "tech", "blockchain", "software", "τεχνολογία"],
    "Health": ["health", "covid", "disease", "hospital", "υγεία", "ιός", "εμβόλιο"],
    "World": ["conflict", "war", "un", "israel", "russia", "world", "παγκόσμιος"],
    "Sports": ["match", "team", "goal", "league", "πρωτάθλημα", "ομάδα", "ποδόσφαιρο"],
}

# --------- MongoDB setup ---------
MONGO_URL = "mongodb://172.25.240.1:27017/"
DB_NAME = "news_database"
COLL_NAME = "articles"

_client = MongoClient(MONGO_URL)
_db = _client[DB_NAME]
collection = _db[COLL_NAME]

# Δημιουργία βασικών index 
try:
    collection.create_index([("link", ASCENDING)], unique=True, name="uniq_link")
    collection.create_index([("hash", ASCENDING)], name="idx_hash")
    collection.create_index([("published_date", ASCENDING)], name="idx_published_date")
except errors.PyMongoError as e:
    logger.warning(f"Index creation warning: {e}")


# --------- Helpers ---------
def classify_category(text: str) -> str:
    t = (text or "").lower()
    for category, keywords in CATEGORIES.items():
        if any(kw in t for kw in keywords):
            return category
    return "General"


def compute_md5(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _extract_raw_html(entry) -> str:
    """Προτιμά entry.content[0].value → entry.summary → ''."""
    try:
        if "content" in entry and entry.content:
            return entry.content[0].value or ""
        if "summary" in entry:
            return entry.summary or ""
    except Exception:
        pass
    return ""


def _pick_entry_date_iso(entry) -> str:
    """
    Προτεραιότητα: published_parsed → updated_parsed → published → updated → now.
    Επιστρέφει ΠΑΝΤΑ ISO σε UTC.
    """
    for key in ("published_parsed", "updated_parsed", "published", "updated"):
        if key in entry and entry[key]:
            return normalize_published_date(entry[key])
    return normalize_published_date(datetime.now(timezone.utc))


# --------- Core crawling ---------
def crawl(limit_per_feed: int = 15, min_content_len: int = 400) -> None:
    """
    Διατρέχει όλα τα FEEDS, καθαρίζει περιεχόμενο/ημερομηνίες και
    εισάγει/ενημερώνει άρθρα στη Mongo με dedup (link/hash).
    """
    logger.info(" Ξεκίνησε το crawling ειδήσεων...")
    new_articles: List[Dict[str, Any]] = []
    inserted = updated = skipped = 0

    for source_name, feed_url in FEEDS.items():
        logger.info(f"📡 Crawling {source_name} ...")
        try:
            feed = feedparser.parse(feed_url)
            entries = getattr(feed, "entries", []) or []
            if not entries:
                logger.warning(f"⚠️ Κενό feed: {source_name}")
                continue

            for entry in entries[:limit_per_feed]:
                # Πεδία
                title_raw = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                raw_html = _extract_raw_html(entry)

                # Καθαρισμός
                title = clean_html(title_raw)
                content = clean_html(raw_html)
                published_date = _pick_entry_date_iso(entry)

                # Ποιότητα
                if not title or not link or len(content) < min_content_len:
                    skipped += 1
                    continue

                category = classify_category(f"{title} {content} {link}")
                content_hash = compute_md5(content)

                doc = {
                    "title": title,
                    "link": link,
                    "published_date": published_date,  # ISO UTC
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "content": content,
                    "category": category,
                    "source": source_name,
                    "hash": content_hash,
                }

                # Insert / Update (με βάση link ή διαφοροποίηση hash)
                try:
                    existing = collection.find_one(
                        {"$or": [{"link": doc["link"]}, {"hash": doc["hash"]}]},
                        projection={"_id": 1, "hash": 1},
                    )
                    if not existing:
                        collection.insert_one(doc)
                        inserted += 1
                        new_articles.append(doc)
                    elif existing.get("hash") != doc["hash"]:
                        collection.update_one({"_id": existing["_id"]}, {"$set": doc})
                        updated += 1
                    else:
                        skipped += 1
                except errors.PyMongoError as e:
                    logger.error(f"❌ Mongo error ({source_name}): {e}")
                    skipped += 1

            logger.info(f"✅ {source_name}: inserted={inserted}, updated={updated}, skipped={skipped}")

        except Exception as e:
            logger.error(f"❌ Σφάλμα κατά το crawling του {source_name}: {e}")

    total = inserted + updated
    if total:
        logger.info(f"Ολοκληρώθηκε το crawling! Νέα/ενημερωμένα άρθρα: {total} (inserted={inserted}, updated={updated}).")
    else:
        logger.warning("⚠️ Δεν βρέθηκαν νέα/ενημερωμένα άρθρα.")


# --------- Optional CLI entrypoint ---------
if __name__ == "__main__":
    crawl()