import requests
from .logits import get_letter_probs
from .prompt import build_prompt

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
TOP_K = 20

# Una sola connessione riusata per tutte le richieste
session = requests.Session()


def request_logprobs(messages):
    payload = {
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": TOP_K,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    response = session.post(SERVER_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def classify_document(document, categories, descriptions=None):
    messages, used_letters = build_prompt(document, categories, descriptions)
    data = request_logprobs(messages)
    top_logprobs = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return get_letter_probs(top_logprobs, used_letters)
