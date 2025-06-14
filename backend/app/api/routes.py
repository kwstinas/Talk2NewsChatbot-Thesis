# app/api/routes.py
from fastapi import APIRouter
from app.chatbot.rag import generate_contextual_answer

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "Talk2News Chatbot API is up and running!"}

@router.post("/ask")
async def ask_question(question: str, category: str = None):
    """
    Λαμβάνει μια ερώτηση και μια προαιρετική κατηγορία και επιστρέφει την απάντηση από το μοντέλο.
    """
    answer = generate_contextual_answer(question, category)
    return {"answer": answer}
