# app/chatbot/llm.py
from llama_cpp import Llama

# ➔ Πού είναι το μοντέλο
# app/chatbot/llm.py

# app/chatbot/llm.py

LLAMA_MODEL_PATH = "/home/kwstinas/Projects/Talk2News-Chatbot/models/llama-2-7b-chat.Q4_K_M.gguf"



# ➔ Συνάρτηση που φορτώνει το μοντέλο
def load_llm():
    llm = Llama(
        model_path=LLAMA_MODEL_PATH,
        f16_kv=True  # Επιτρέπει καλύτερη απόδοση μνήμης
    )
    return llm

# ➔ Συνάρτηση για να πάρουμε απάντηση
def generate_answer(llm, prompt: str) -> str:
    response = llm(
        prompt=prompt,
        max_tokens=1024,  
        temperature=0.7,  # Θερμοκρασία για τη δημιουργία πιο φυσικών απαντήσεων
        top_p=0.9,        # Επιλογή πιθανοτήτων για την απάντηση
        stop=["</s>"]     # Σταματά όταν φτάσει στο </s> (αν χρειάζεται για τον τερματισμό)
    )
    return response["choices"][0]["text"]
