# backend/app/chatbot/llm.py
import os
import logging
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Ρύθμιση μοντέλου 
LLAMA_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    "/home/kwstinas/Projects/Talk2News-Chatbot/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
)

# ✅ Ρυθμίσεις φόρτωσης
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "4096"))
LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))  # 0 = CPU
LLAMA_CHAT_FORMAT = os.getenv("LLAMA_CHAT_FORMAT", "llama-3")   

_llm = None  # global cache


def load_llm():
    """
    Φορτώνει και κάνει cache το Llama. Αν αποτύχει, επιστρέφει None.
    """
    global _llm
    if _llm is not None:
        return _llm
    try:
        if not os.path.exists(LLAMA_MODEL_PATH):
            raise FileNotFoundError(f"Model path not found: {LLAMA_MODEL_PATH}")

        _llm = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=LLAMA_N_CTX,
            n_gpu_layers=LLAMA_N_GPU_LAYERS,
            chat_format=LLAMA_CHAT_FORMAT,  
            f16_kv=True,
        )
        logger.info(
            "✅ Llama loaded | path=%s | chat_format=%s | n_ctx=%d | n_gpu_layers=%d",
            LLAMA_MODEL_PATH, LLAMA_CHAT_FORMAT, LLAMA_N_CTX, LLAMA_N_GPU_LAYERS
        )
    except Exception as e:
        logger.error("⚠️ LLama model failed to load: %s", e)
        _llm = None
    return _llm


def generate_answer(llm, prompt: str) -> str:
    """
    Παίρνει ΕΝΑ concatenated prompt (όπως το φτιάχνεις στο RAG)
    και το περνάει ως system+user messages, με ασφαλή καθαρισμό UTF-8
    και συντηρητικές παραμέτρους για λιγότερο "φλύαρες" απαντήσεις.
    """
    logger.info("Generating answer with Llama model...")
    try:
        if llm is None:
            return "Σφάλμα: Το μοντέλο δεν φορτώθηκε (έλεγξε LLAMA_MODEL_PATH ή το αρχείο .gguf)."

        # ---- Ασφαλής μετατροπή & καθάρισμα UTF-8 (αποφυγή surrogate errors) ----
        if not isinstance(prompt, str):
            prompt = str(prompt)
        prompt = prompt.encode("utf-8", "ignore").decode("utf-8", "ignore").strip()

        # ---- System/User split με markers (fallback: όλο ως user) ----
        sys_part = ""
        user_part = prompt
        for sep in ("[ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ]", "[USER QUESTION]"):
            if sep in prompt:
                left, right = prompt.split(sep, 1)
                sys_part = left.strip()
                user_part = right.strip()
                break

        messages = []
        if sys_part:
            messages.append({"role": "system", "content": sys_part})
        else:
            messages.append({
                "role": "system",
                "content": "You are a factual news assistant. Answer concisely and only from the provided content."
            })
        messages.append({"role": "user", "content": user_part})

        # ---- Κλήση στο chat completion με πιο “σφιχτές” παραμέτρους ----
        resp = llm.create_chat_completion(
            messages=messages,
            temperature=0.15,
            top_p=0.85,
            top_k=30,
            max_tokens=350,
            repeat_penalty=1.2,
            stop=["</s>", "[END]", "[ΤΕΛΟΣ]"],
        )

        # ---- Ανάγνωση απάντησης ----
        choice = resp.get("choices", [{}])[0]
        text = (
            choice.get("message", {}).get("content")
            or choice.get("text")
            or ""
        )

        # ---- Τελικό καθάρισμα ----
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore").strip()
        return text

    except Exception as e:
        logger.error("Error in generating answer: %s", e)
        return ""
