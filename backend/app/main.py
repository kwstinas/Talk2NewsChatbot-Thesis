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
from .crawler.crawler import crawl
from .api.routes import router as api_router
from .chatbot.vectorstore import load_vectorstore
from fastapi import Request
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # --- STARTUP ---
    print("Initial startup - performing sync crawl and update...")
    new_articles_count, _ = _crawl_only()
    
    if new_articles_count is not None and new_articles_count > 0:
        print(f"Starting sync vectorstore update with {new_articles_count} new articles...")
        _async_update_vectorstore()
    else:
        print("No new articles on startup - skipping initial update")
    
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("Σφάλμα: Το vectorstore δεν είναι διαθέσιμο κατά την εκκίνηση του server!")
    else:
        doc_count = len(vectorstore.docstore._dict)
        print(f"Vectorstore loaded! ({doc_count} documents)")

    scheduler.add_job(
        _crawl_and_async_update,
        "interval",
        hours=1,
        id="crawl-async-update-hourly",
        replace_existing=True
    )
    scheduler.add_job(
        _safe_full_incremental_update,
        "cron",
        hour=3, minute=0,
        id="faiss-full-incremental-daily",
        replace_existing=True,
    )
    scheduler.start()
    print("Scheduler ξεκίνησε! Fast crawl + async FAISS update κάθε 1 ώρα")

    yield

    # --- SHUTDOWN ---
    scheduler.shutdown()
    print("Scheduler σταμάτησε.")

# FastAPI app
app = FastAPI(
    title="Talk2News Chatbot",
    version="1.0.0",
    description="Ένα chatbot που απαντάει σε ερωτήσεις σχετικά με νέα και ειδήσεις.",
    lifespan=lifespan
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
        
        #Αν ο crawler επιστρέφει None, θεωρούμε 0
        if new_articles_count is None:
            new_articles_count = 0
            print("Crawler returned None - treating as 0 new articles")
        
        crawl_duration = (datetime.now() - start_time).total_seconds()
        
        print(f"Crawl completed in {crawl_duration:.1f}s - {new_articles_count} new articles")
        return new_articles_count, crawl_duration
        
    except Exception as e:
        print(f"Crawl failed: {e}")
        return 0, 0  # Σε περίπτωση σφάλματος, επέστρεψε 0

def _async_update_vectorstore():
    """
    Ασύγχρονο incremental update - δεν μπλοκάρει το main thread.
    """
    print("Starting ASYNC vectorstore update...")
    try:
        # Μόνο άρθρα των τελευταίων 2 ωρών για speed
        incremental_update_vectorstore(hours=2)
        
        # Reload vectorstore
        vs = reload_vectorstore()
        if vs:
            doc_count = len(vs.docstore._dict)
            print(f"ASYNC vectorstore update completed! ({doc_count} documents)")
        else:
            print("ASYNC vectorstore update failed")
    except Exception as e:
        print(f"ERROR in async vectorstore update: {e}")

def _crawl_and_async_update():
    """
    Crawl + async vectorstore update αν βρεθούν άρθρα.
    """
    new_articles_count, crawl_duration = _crawl_only()
    
    # Explicit check για None και > 0
    if new_articles_count is not None and new_articles_count > 0:
        print(f" Found {new_articles_count} new articles - starting async FAISS update...")
        
        # Ασύγχρονο update - δεν περιμένει
        update_thread = threading.Thread(target=_async_update_vectorstore)
        update_thread.daemon = True
        update_thread.start()
        
        print("Async FAISS update started in background...")
    else:
        print("No new articles - skipping FAISS update")

def _safe_full_incremental_update():
    """Ασφαλής full incremental update."""
    try:
        print(f"[{datetime.now()}] Starting nightly FULL FAISS update...")
        incremental_update_vectorstore(hours=24)
        
        vs = reload_vectorstore()
        if vs:
            doc_count = len(vs.docstore._dict)
            print(f"Nightly FULL FAISS update completed! ({doc_count} documents)")
    except Exception as e:
        print(f"ERROR in nightly FAISS update: {e}")

# οδηγεί σε /frontend στη ρίζα του repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.middleware("http")
async def add_cache_control_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.endswith(('.jsx', '.js')):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response