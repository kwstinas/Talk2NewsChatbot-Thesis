# app/main.py

from fastapi import FastAPI
from app.crawler.crawler import crawl
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router  # Εισάγουμε το router από το api.routes
import pytz  # Εισάγουμε το pytz για τη ζώνη ώρας
from pydantic import BaseModel
from app.chatbot.rag import generate_contextual_answer  # Εισάγουμε τη συνάρτηση για απάντηση

# Δημιουργία του FastAPI app
app = FastAPI(
    title="Talk2News Chatbot",
    version="1.0.0",
    description="Ένα chatbot που απαντάει σε ερωτήσεις σχετικά με νέα και ειδήσεις."
)

# Δημιουργία του scheduler με ζώνη ώρας από την pytz
scheduler = BackgroundScheduler(timezone=pytz.timezone("UTC"))

# Συνάρτηση για την υποβολή ερωτήσεων
class Question(BaseModel):
    question: str
    category: str = None  # Προαιρετική κατηγορία για φιλτράρισμα άρθρων

@app.post("/ask/")
async def ask_question(question: Question):
    """
    Route για την υποβολή ερωτήσεων και την επιστροφή της απάντησης από το chatbot.
    """
    # Καλούμε την generate_contextual_answer για να πάρουμε την απάντηση
    answer = generate_contextual_answer(question.question, question.category)
    return {"answer": answer}

@app.on_event("startup")
def startup_event():
    """
    Συνάρτηση για να ξεκινήσει το πρώτο crawling κατά την εκκίνηση του server
    και να ρυθμίσει τον scheduler να τρέχει το crawling κάθε ώρα.
    """
    # Κάνε το πρώτο crawling μόλις ξεκινήσει ο server
    crawl()

    # Ρύθμισε τον scheduler να τρέχει το crawl κάθε 1 ώρα
    scheduler.add_job(crawl, "interval", hours=1)
    scheduler.start()

    print("Scheduler ξεκίνησε! Θα γίνεται crawling κάθε 1 ώρα.")

@app.on_event("shutdown")
def shutdown_event():
    """
    Συνάρτηση για να σταματήσει το scheduler όταν ο server σταματήσει.
    """
    scheduler.shutdown()
    print("Scheduler σταμάτησε.")

# Εισαγωγή των routes του API από το api.routes
app.include_router(api_router, prefix="/api")
scheduler.add_job(crawl, 'interval', hours=1, max_instances=1, misfire_grace_time=30)
