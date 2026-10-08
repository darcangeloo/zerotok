import os
import requests
from .logits import get_letter_probs
from .prompt import build_prompt, LETTERS

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
TOP_K = 20

# "llamacpp" (default, la baseline) oppure "branching"
BACKEND = os.environ.get("ZEROTOK_BACKEND", "llamacpp")
HEAD_PATH = os.environ.get("ZEROTOK_HEAD", "head/head.pt")
HEAD_DIM = 512  # deve essere lo stesso new_dim usato in training

# Una sola connessione riusata per tutte le richieste
session = requests.Session()

# (testa, temperatura): si caricano una volta sola, alla prima richiesta branching
_head = None


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


def classify_llamacpp(document, categories, descriptions=None):
    messages, used_letters = build_prompt(document, categories, descriptions)
    data = request_logprobs(messages)
    top_logprobs = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return get_letter_probs(top_logprobs, used_letters)


def load_head():
    global _head
    if _head is None:
        from .head import load_head as load_head_from_file
        _head = load_head_from_file(HEAD_PATH)
    return _head


def classify_branching(document, categories, descriptions=None):
    import torch
    from .branching import get_choice_embeddings

    head, temperature = load_head()
    choice_emb, answer_emb = get_choice_embeddings(document, categories, descriptions)

    # Una sola richiesta, quindi nessun padding: tutte le scelte sono reali
    mask = torch.ones(1, len(categories), dtype=torch.bool)
    with torch.no_grad():
        logits = head(choice_emb.unsqueeze(0), answer_emb.unsqueeze(0), mask)[0]
    probs = torch.softmax(logits / temperature, dim=-1).tolist()

    # Stesso formato della baseline, cosi app.py ed evaluation.py non cambiano
    probs_by_letter = {}
    for i, p in enumerate(probs):
        probs_by_letter[LETTERS[i]] = p
    return probs_by_letter


def classify_document(document, categories, descriptions=None):
    if BACKEND == "branching":
        return classify_branching(document, categories, descriptions)
    return classify_llamacpp(document, categories, descriptions)