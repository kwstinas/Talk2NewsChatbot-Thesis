# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

from .crawler.crawler import crawl
from .api.routes import router as api_router
from .chatbot.vectorstore import load_vectorstore


# Δημιουργία του FastAPI app
app = FastAPI(
    title="Talk2News Chatbot",
    version="1.0.0",
    description="Ένα chatbot που απαντάει σε ερωτήσεις σχετικά με νέα και ειδήσεις."
)

# CORS αν χρειαστεί
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include τα routes σου
app.include_router(api_router, prefix="/api")

# Scheduler για το crawling
scheduler = BackgroundScheduler(timezone=pytz.timezone("UTC"))

@app.on_event("startup")
def startup_event():
    """
    Συνάρτηση για να ξεκινήσει το πρώτο crawling κατά την εκκίνηση του server
    και να ρυθμίσει τον scheduler να τρέχει το crawling κάθε ώρα.
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
