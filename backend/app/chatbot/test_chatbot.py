# backend/app/chatbot/test_chatbot.py

from app.chatbot.rag import generate_contextual_answer  # Εισάγουμε τη συνάρτηση για απάντηση
from pprint import pprint

def test_chatbot():
    # Ερώτηση από τον χρήστη
    question = input("Βάλε την ερώτηση σου: ")

    # (Προαιρετικά) καθορισμός κατηγορίας (π.χ. "Politics", "Technology" κλπ.)
    category = input("Πληκτρολόγησε την κατηγορία (ή άφησέ το κενό): ")

    # Κλήση της συνάρτησης για να πάρουμε την απάντηση
    answer = generate_contextual_answer(question, category if category else None)

    # Εκτύπωση της απάντησης
    print("\nΑπάντηση: ")
    pprint(answer)  # Εμφανίζουμε την απάντηση με καλύτερη μορφοποίηση

if __name__ == "__main__":
    test_chatbot()
