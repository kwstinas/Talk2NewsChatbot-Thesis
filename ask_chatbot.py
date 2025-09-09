import requests
import sys
# Ορίζουμε το URL του backend API
url = "http://127.0.0.1:8000/api/ask"

def ask_question(query):
    """
    Στέλνει ερώτηση στο backend API και επιστρέφει την απάντηση.
    """
    # πάντα στέλνουμε κενό string ως category για συμβατότητα
    payload = {"question": query.strip(), "category": ""}

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            if 'answer' in response_data:
                return response_data['answer']
            else:
                return "Δεν βρέθηκε έγκυρη απάντηση από το API."
        else:
            return f"Error: {response.status_code}, {response.text}"

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"

import sys

def safe_print(s: str):
    try:
        print(s)
    except UnicodeEncodeError:
        cleaned = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
        print(cleaned)

# ...
if __name__ == "__main__":
    print("Welcome to Talk2News Chatbot! Type 'exit' to quit.")
    while True:
        question = input("Enter your question: ")
        if question.lower() == "exit":
            print("Exiting Chatbot. Goodbye!")
            break
        answer = ask_question(question)
        safe_print("Answer: " + (answer or ""))

