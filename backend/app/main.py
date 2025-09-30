# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import os

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

# Scheduler για crawling

scheduler = BackgroundScheduler(timezone=pytz.timezone("UTC"))

@app.on_event("startup")
def startup_event():
    """
    Στο startup:
    - Τρέχουμε ένα αρχικό crawl
    - Φορτώνουμε το vectorstore 
    - Ξεκινάμε scheduler ανά 1 ώρα
    """
    crawl()
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("Σφάλμα: Το vectorstore δεν είναι διαθέσιμο κατά την εκκίνηση του server!")
    else:
        print("Το vectorstore φορτώθηκε επιτυχώς!")

    scheduler.add_job(crawl, "interval", hours=1)
    scheduler.start()
    print("Scheduler ξεκίνησε! Θα γίνεται crawling κάθε 1 ώρα.")

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
    print("Scheduler σταμάτησε.")

# ΜΕΤΑ (οδηγεί σε /frontend στη ρίζα του repo)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")


# Σερβίρουμε τα στατικά αρχεία (JSX/CSS/favicon) 
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Η ρίζα σερβίρει το index.html
@app.get("/")
async def root_page():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)
