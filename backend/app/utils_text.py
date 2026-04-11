# backend/app/utils_text.py
import re, html, time
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser, tz

ATHENS = tz.gettz("Europe/Athens")
UTC = tz.gettz("UTC")

DOMAIN_SOURCE_MAPPING = {
    "theverge.com": "The Verge",
    "techradar.com": "TechRadar",
    "apnews.com": "Associated Press",
    "newsbeast.gr": "Newsbeast",
    "naftemporiki.gr": "Naftemporiki",
    "theguardian.com": "The Guardian",
    "guardian.com": "The Guardian",
    "techcrunch.com": "TechCrunch",
    "skai.gr": "SKAI",
    "in.gr": "In.gr",
    "tovima.gr": "To Vima",
    "documentonews.gr": "Documento",
    "greekreporter.com": "Greek Reporter",
    "abcnews.go.com": "ABC News",
    "npr.org": "NPR News",
    "eleftherostypos.gr": "Eleftheros Typos",
    "tanea.gr": "TaNea",
}

def get_source_from_link(link: str) -> str:
    """Εξάγει πηγή από URL"""
    if not link:
        return ""
    link_lower = link.lower()
    for domain, source_name in DOMAIN_SOURCE_MAPPING.items():
        if domain in link_lower:
            return source_name
    return ""

def normalize_published_date(value) -> str:
    """
    Δέχεται string | datetime | time.struct_time και γυρνά πάντα ISO UTC string.
    """
    if value is None:
        dt = datetime.now(UTC)
        return dt.isoformat()
    if isinstance(value, time.struct_time):
        dt = datetime(*value[:6])
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = parser.parse(str(value), fuzzy=True)

    if dt.tzinfo is None:
        # Αν δεν έχει ζώνη, υποθέτουμε Αθήνα (ή βάλε UTC αν προτιμάς)
        dt = dt.replace(tzinfo=ATHENS)
    return dt.astimezone(UTC).isoformat()

def strip_surrogates(s: str) -> str:
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore")
    # πετάει invalid surrogate code points
    return s.encode("utf-8", "ignore").decode("utf-8", "ignore")

def clean_html(raw_html: str) -> str:
    """
    Αφαιρεί tags/scripts/styles, κάνει unescape & συμπίεση κενών.
    Επιστρέφει «στεγνό» κείμενο.
    """
    raw_html = strip_surrogates(str(raw_html))
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text