# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .chatbot.build_vectorstore import build_vectorstore
from .chatbot.build_vectorstore import incremental_update_vectorstore
from .chatbot.vectorstore import reload_vectorstore, get_vectorstore_info
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import os
from datetime import datetime, timedelta
import threading


from pymongo import MongoClient
from datetime import datetime, timedelta


from .crawler.crawler import crawl
from .api.routes import router as api_router
from .chatbot.vectorstore import load_vectorstore

# FastAPI app
app = FastAPI(
    title="Talk2News Chatbot",
    version="1.0.0",
    description="Ένα chatbot που απαντάει σε ερωτήσεις σχετικά με νέα και ειδήσεις."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes 
app.include_router(api_router, prefix="/api")

# Scheduler για crawling και vectorstore updates
scheduler = BackgroundScheduler(timezone=pytz.timezone("UTC"))

def _crawl_only():
    """
    Crawl μόνο - με proper error handling.
    Επιστρέφει: (new_articles_count, crawl_duration)
    """
    start_time = datetime.now()
    print(f"[{start_time}] Starting fast crawl...")
    
    try:
        new_articles_count = crawl()
        
        # 🔧 FIX: Αν ο crawler επιστρέφει None, θεωρούμε 0
        if new_articles_count is None:
            new_articles_count = 0
            print("⚠️ Crawler returned None - treating as 0 new articles")
        
        crawl_duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Crawl completed in {crawl_duration:.1f}s - {new_articles_count} new articles")
        return new_articles_count, crawl_duration
        
    except Exception as e:
        print(f"❌ Crawl failed: {e}")
        return 0, 0  # Σε περίπτωση σφάλματος, επέστρεψε 0

def _async_update_vectorstore():
    """
    Ασύγχρονο incremental update - δεν μπλοκάρει το main thread.
    """
    print("🔄 Starting ASYNC vectorstore update...")
    try:
        # Μόνο άρθρα των τελευταίων 2 ωρών για speed
        incremental_update_vectorstore(hours=2)
        
        # Reload vectorstore
        vs = reload_vectorstore()
        if vs:
            doc_count = len(vs.docstore._dict)
            print(f"✅ ASYNC vectorstore update completed! ({doc_count} documents)")
        else:
            print("⚠️ ASYNC vectorstore update failed")
    except Exception as e:
        print(f"❌ ERROR in async vectorstore update: {e}")

def _crawl_and_async_update():
    """
    Crawl + async vectorstore update αν βρεθούν άρθρα.
    """
    new_articles_count, crawl_duration = _crawl_only()
    
    # 🔧 FIX: Explicit check για None και > 0
    if new_articles_count is not None and new_articles_count > 0:
        print(f"📥 Found {new_articles_count} new articles - starting async FAISS update...")
        
        # Ασύγχρονο update - δεν περιμένει
        update_thread = threading.Thread(target=_async_update_vectorstore)
        update_thread.daemon = True
        update_thread.start()
        
        print("⚡ Async FAISS update started in background...")
    else:
        print("ℹ️ No new articles - skipping FAISS update")

@app.on_event("startup")
def startup_event():
    # Αρχικό crawl (σύγχρονο για να είμαστε σίγουροι)
    print("🚀 Initial startup - performing sync crawl and update...")
    new_articles_count, _ = _crawl_only()
    
    # 🔧 FIX: Safe check για startup
    if new_articles_count is not None and new_articles_count > 0:
        print(f"🔄 Starting sync vectorstore update with {new_articles_count} new articles...")
        _async_update_vectorstore()  # Σύγχρονο στο startup
    else:
        print("ℹ️ No new articles on startup - skipping initial update")
    
    # Αρχική φόρτωση vectorstore
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("Σφάλμα: Το vectorstore δεν είναι διαθέσιμο κατά την εκκίνηση του server!")
    else:
        doc_count = len(vectorstore.docstore._dict)
        print(f"✅ Vectorstore loaded! ({doc_count} documents)")

    # jobs - crawl με async update
    scheduler.add_job(
        _crawl_and_async_update,
        "interval", 
        hours=1, 
        id="crawl-async-update-hourly", 
        replace_existing=True
    )

    # ΜΙΑ φορά την ημέρα (03:00 UTC) FULL incremental FAISS update
    scheduler.add_job(
        _safe_full_incremental_update,
        "cron",
        hour=3, minute=0,
        id="faiss-full-incremental-daily",
        replace_existing=True,
    )

    scheduler.start()
    print("Scheduler ξεκίνησε! Fast crawl + async FAISS update κάθε 1 ώρα")

def _safe_full_incremental_update():
    """Ασφαλής full incremental update (σύγχρονο - νυχτερινό)."""
    try:
        print(f"[{datetime.now()}] Starting nightly FULL FAISS update...")
        incremental_update_vectorstore(hours=24)
        
        vs = reload_vectorstore()
        if vs:
            doc_count = len(vs.docstore._dict)
            print(f"✅ Nightly FULL FAISS update completed! ({doc_count} documents)")
    except Exception as e:
        print(f"❌ ERROR in nightly FAISS update: {e}")

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
    print("Scheduler σταμάτησε.")

# Health check endpoint
@app.get("/api/vectorstore-status")
async def vectorstore_status():
    return get_vectorstore_info()

# ΜΕΤΑ (οδηγεί σε /frontend στη ρίζα του repo)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

def _debug_check_recent_articles():
    """Έλεγχος για άρθρα των τελευταίων 2 ημερών."""
    client = MongoClient("mongodb://172.25.240.1:27017/")
    db = client["news_database"]
    collection = db["articles"]
    
    cutoff = datetime.now() - timedelta(days=2)
    cutoff_iso = cutoff.isoformat()
    
    recent_count = collection.count_documents({"published_date": {"$gte": cutoff_iso}})
    print(f"🔍 DEBUG: Άρθρα των τελευταίων 2 ημερών: {recent_count}")
    
    # Δείξε μερικά πρόσφατα άρθρα
    recent_articles = list(collection.find(
        {"published_date": {"$gte": cutoff_iso}}
    ).sort("published_date", -1).limit(5))
    
    for art in recent_articles:
        print(f"  - {art.get('title')} | {art.get('published_date')} | {art.get('source')}")

def _debug_check_new_articles():
    """Έλεγχος για πραγματικά νέα άρθρα."""
    from pymongo import MongoClient
    from datetime import datetime, timedelta
    
    client = MongoClient("mongodb://172.25.240.1:27017/")
    db = client["news_database"]
    collection = db["articles"]
    
    # Άρθρα των τελευταίων 7 ημερών
    cutoff = datetime.now() - timedelta(days=7)
    cutoff_iso = cutoff.isoformat()
    
    recent_articles = list(collection.find({
        "$or": [
            {"published_date": {"$gte": cutoff_iso}},
            {"fetched_at": {"$gte": cutoff_iso}}
        ]
    }).sort("_id", -1).limit(20))
    
    print(f"🔍 RECENT ARTICLES (last 7 days): {len(recent_articles)}")
    for art in recent_articles[:5]:  # Πρώτα 5 μόνο
        title = art.get('title', 'No title')[:60]
        pub_date = art.get('published_date', 'No date')
        fetched = art.get('fetched_at', 'No fetch')
        source = art.get('source', 'Unknown')
        print(f"  - '{title}'...")
        print(f"    Pub: {pub_date} | Fetch: {fetched}")
        print(f"    Source: {source}")

def _debug_check_actual_articles():
    """Έλεγχος για ΠΡΑΓΜΑΤΙΚΑ πρόσφατα άρθρα."""
    from pymongo import MongoClient
    from datetime import datetime, timedelta
    
    client = MongoClient("mongodb://172.25.240.1:27017/")
    db = client["news_database"]
    collection = db["articles"]
    
    # Άρθρα των τελευταίων 3 ημερών
    cutoff = datetime.now() - timedelta(days=3)
    cutoff_iso = cutoff.isoformat()
    
    print("🔍 CHECKING FOR RECENT ARTICLES (last 3 days):")
    
    # Έλεγχος με fetched_at (πιο αξιόπιστο)
    recent_by_fetched = list(collection.find({
        "fetched_at": {"$gte": cutoff_iso}
    }).sort("fetched_at", -1).limit(10))
    
    print(f"📥 Articles by fetched_at: {len(recent_by_fetched)}")
    for art in recent_by_fetched:
        title = art.get('title', 'No title')[:70]
        fetched = art.get('fetched_at', 'No fetch')
        pub_date = art.get('published_date', 'No pub date')
        source = art.get('source', 'Unknown')
        print(f"  - '{title}'...")
        print(f"    Fetched: {fetched}")
        print(f"    Published: {pub_date}")
        print(f"    Source: {source}")
        print()
    
    # Έλεγχος για Greece/Politics topics
    greece_politics = list(collection.find({
        "$or": [
            {"title": {"$regex": "greece|ελλάδα|πολιτική|politics", "$options": "i"}},
            {"content": {"$regex": "greece|ελλάδα|πολιτική|politics", "$options": "i"}},
            {"category": "Politics"}
        ],
        "fetched_at": {"$gte": cutoff_iso}
    }).limit(5))
    
    print(f"🇬🇷 Greece/Politics articles: {len(greece_politics)}")
    for art in greece_politics:
        title = art.get('title', 'No title')[:70]
        print(f"  - '{title}'...")