# app/api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from ..chatbot.rag import generate_contextual_answer


router = APIRouter()


@router.get("/")
async def read_root():
    return {"message": "Talk2News Chatbot API is up and running!"}


class Question(BaseModel):
    question: str
    category: str = None  # Προαιρετική κατηγορία


@router.post("/ask")
async def ask_question(question: Question):
    """
    Λαμβάνει μια ερώτηση και μια προαιρετική κατηγορία και επιστρέφει την απάντηση από το μοντέλο.
    """
    answer = generate_contextual_answer(question.question, question.category)
    return {"answer": answer}
