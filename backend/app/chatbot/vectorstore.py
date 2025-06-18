# app/chatbot/vectorstore.py

import os
from langchain.vectorstores import FAISS  # Χρησιμοποιούμε το σωστό FAISS από την βιβλιοθήκη langchain
from langchain.embeddings import HuggingFaceEmbeddings

# Ο φάκελος όπου θα αποθηκεύεται το FAISS vectorstore
SAVE_PATH = "faiss_index"

def create_vectorstore(texts, metadatas):
    """
    Δημιουργεί το FAISS vectorstore από τα κείμενα και τα metadata των άρθρων.
    """
    # Δημιουργία του HuggingFace embeddings model
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Δημιουργία του FAISS vectorstore
    vectorstore = FAISS.from_texts(texts, embedder, metadatas=metadatas)
    
    # Αποθήκευση του FAISS vectorstore στο τοπικό σύστημα
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    vectorstore.save_local(SAVE_PATH)
    print(f"FAISS vectorstore αποθηκεύθηκε στο {SAVE_PATH}.")

def load_vectorstore():
    """
    Φορτώνει το FAISS vectorstore από το αποθηκευμένο path.
    """
    # Ελέγξτε αν το FAISS vectorstore υπάρχει πριν το φορτώσετε
    if os.path.exists(SAVE_PATH):
        print(f"Φόρτωση του FAISS vectorstore από το {SAVE_PATH}.")
        return FAISS.load_local(SAVE_PATH, HuggingFaceEmbeddings())
    else:
        print(f"Το FAISS vectorstore δεν υπάρχει στο {SAVE_PATH}.")
        return None
