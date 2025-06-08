from datetime import datetime, timedelta, timezone
from app.chatbot.vectorstore import load_vectorstore
from app.chatbot.llm import load_llm, generate_answer

# 🔹 Φόρτωμα vectorstore και LLM
vectorstore = load_vectorstore()
llm = load_llm()

# 🔹 Helper: φιλτράρισμα πρόσφατων εγγράφων
def filter_recent_documents(documents, days=30, desired_category=None):
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    filtered_docs = []

    for doc in documents:
        metadata = doc.metadata
        published_date_str = metadata.get("published_date")
        category = metadata.get("category", "General").lower()

        if not published_date_str:
            continue

        try:
            published_date = datetime.fromisoformat(published_date_str)
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=timezone.utc)

            if published_date >= cutoff_date:
                # Αν έχει ζητηθεί κατηγορία, φιλτράρουμε περαιτέρω
                if desired_category:
                    if category == desired_category.lower():
                        filtered_docs.append(doc)
                else:
                    filtered_docs.append(doc)
        except Exception as e:
            print(f"⚠️ Σφάλμα parsing ημερομηνίας: {published_date_str} -> {e}")
            continue

    print(f"🧪 Νέα φίλτρα άρθρων: {len(filtered_docs)} / {len(documents)}")
    return filtered_docs

# 🔹 Βασική RAG λειτουργία
def generate_contextual_answer(user_query, category: str = None):
    if not user_query.strip():
        return " Δεν δόθηκε έγκυρη ερώτηση."

    relevant_docs = vectorstore.similarity_search(user_query, k=15)

    if not relevant_docs:
        return " Δεν βρέθηκαν σχετικά άρθρα."

    recent_docs = filter_recent_documents(relevant_docs, days=30, desired_category=category)

    if not recent_docs:
        return " Δεν βρέθηκαν πρόσφατες ειδήσεις για το θέμα."

    context = "\n\n".join(doc.page_content for doc in recent_docs)

    # 🔹 Βελτιωμένο δημοσιογραφικό prompt
    prompt = f"""
Είσαι ένας έμπειρος δημοσιογράφος που γράφει σε βάθος, με εγκυρότητα και σαφήνεια. Ακολουθούν αποσπάσματα από πρόσφατα άρθρα ειδήσεων:

{context}

Βάσει των παραπάνω αποσπασμάτων, απάντησε στην ακόλουθη ερώτηση:

Ερώτηση: {user_query}

 ΟΔΗΓΙΕΣ:
- Ανάλυσε την ερώτηση προσεκτικά και απάντησε με πληρότητα και σαφήνεια, σαν να γράφεις άρθρο γνώμης ή ανασκόπηση για ενημερωμένο κοινό.
- Μην εφευρίσκεις πληροφορίες που δεν περιλαμβάνονται στο περιεχόμενο.
- Αν υπάρχουν αντιφατικές πληροφορίες, παρουσίασέ τες και εξήγησε την πιθανή αιτία.
- Αν δεν υπάρχουν επαρκείς πληροφορίες για να απαντήσεις, πες με σαφήνεια: "Δεν υπάρχουν διαθέσιμες πληροφορίες για το συγκεκριμένο ερώτημα αυτή τη στιγμή."

 Η απάντηση πρέπει να είναι καλογραμμένη, σοβαρή και δομημένη, όπως σε ένα άρθρο μέσου ενημέρωσης.

Απάντηση:
"
"""

    return generate_answer(llm, prompt).strip()
