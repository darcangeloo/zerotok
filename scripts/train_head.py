import sys
from pathlib import Path

# Permette di lanciare lo script anche direttamente: aggiungo la root al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import hashlib
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

from src.branching import get_choice_embeddings, model
from src.head import ChoiceHead, save_head

N_TRAIN = 5000
N_VAL = 1000
MAX_STATE_CHARS = 3000
MAX_CHOICES = 10
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4

random.seed(0)
torch.manual_seed(0)


def is_val(state):
    # Divisione per stato: lo stesso stato finisce sempre dalla stessa parte
    return int(hashlib.md5(state.encode("utf-8")).hexdigest(), 16) % 10 == 0


def load_questions(row):
    questions = json.loads(row["questions_json"])
    # La card parla di un oggetto "questions", le righe che ho visto no: gestisco entrambi i casi
    if "questions" in questions:
        questions = questions["questions"]
    return list(questions.values())


def is_usable(question):
    if question.get("type") != "choice" or not isinstance(question.get("criteria"), dict):
        return False
    keys = list(question["criteria"].keys())
    # Servono almeno 2 scelte, non piu di MAX_CHOICES, e la risposta giusta deve esserci
    return 2 <= len(keys) <= MAX_CHOICES and question["label"] in keys


def to_examples(row):
    if len(row["state"]) > MAX_STATE_CHARS:
        return []

    examples = []
    for question in load_questions(row):
        if not is_usable(question):
            continue
        keys = list(question["criteria"].keys())
        # Uso il testo delle opzioni, non le chiavi (option_0, test_1) che rimetterebbero un ordine
        choices = []
        for key in keys:
            choices.append(str(question["criteria"][key]))
        # L'istruzione va nel documento, altrimenti il modello non sa cosa deve decidere
        document = row["state"] + "\n\n" + question["instructions"]
        examples.append((document, choices, keys.index(question["label"])))
    return examples


def build_sets():
    ds = load_dataset("avbiswas/bev-decision", split="train").shuffle(seed=0)
    train, val = [], []
    for row in ds:
        if len(train) >= N_TRAIN and len(val) >= N_VAL:
            break
        if is_val(row["state"]):
            if len(val) < N_VAL:
                val.extend(to_examples(row))
        elif len(train) < N_TRAIN:
            train.extend(to_examples(row))
    return train[:N_TRAIN], val[:N_VAL]


def extract(examples, path):
    # Il backbone e congelato: calcolo gli embedding una volta sola e li riuso a ogni epoca
    if os.path.exists(path):
        print(f"Carico la cache da {path}")
        return torch.load(path)
    cache = []
    start = time.perf_counter()
    for i, (document, choices, label) in enumerate(examples):
        c, a = get_choice_embeddings(document, choices)
        cache.append({"c": c.half(), "a": a.half(), "y": label})
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"{i + 1}/{len(examples)}, {elapsed / (i + 1):.2f} s per esempio")
    torch.save(cache, path)
    return cache


def make_batch(items):
    # Le domande hanno numeri di scelte diversi: riempio con zeri e segno il padding nella maschera
    B = len(items)
    C = max(item["c"].shape[0] for item in items)
    H = items[0]["c"].shape[1]
    choice_emb = torch.zeros(B, C, H)
    mask = torch.zeros(B, C, dtype=torch.bool)
    for i, item in enumerate(items):
        n = item["c"].shape[0]
        choice_emb[i, :n] = item["c"].float()
        mask[i, :n] = True
    answer_emb = torch.stack([item["a"].float() for item in items])
    labels = torch.tensor([item["y"] for item in items])
    return choice_emb, answer_emb, mask, labels


def train_one_epoch(head, optimizer, train_cache):
    head.train()
    # L'ordine cambia a ogni epoca, altrimenti la testa vede sempre i batch nello stesso ordine
    random.shuffle(train_cache)
    total_loss = 0.0
    for i in range(0, len(train_cache), BATCH_SIZE):
        c, a, m, y = make_batch(train_cache[i:i + BATCH_SIZE])
        loss = F.cross_entropy(head(c, a, m), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def accuracy(head, cache):
    head.eval()
    correct = 0
    for i in range(0, len(cache), BATCH_SIZE):
        c, a, m, y = make_batch(cache[i:i + BATCH_SIZE])
        pred = head(c, a, m).argmax(dim=-1)
        correct += (pred == y).sum().item()
    return correct / len(cache)


if __name__ == "__main__":

    train_examples, val_examples = build_sets()
    print(f"train {len(train_examples)}, val {len(val_examples)}")

    train_cache = extract(train_examples, "head/cache_train.pt")
    val_cache = extract(val_examples, "head/cache_val.pt")

    # Livello del caso: se la testa non lo supera, non sta imparando niente
    chance = sum(1 / item["c"].shape[0] for item in val_cache) / len(val_cache)
    print(f"accuratezza a caso sulla val: {chance:.3f}")

    head = ChoiceHead(hidden_dim=model.config.hidden_size, new_dim=512)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    n_batches = math.ceil(len(train_cache) / BATCH_SIZE)

    best = 0.0
    for epoch in range(EPOCHS):
        total_loss = train_one_epoch(head, optimizer, train_cache)
        val_acc = accuracy(head, val_cache)
        print(f"epoca {epoch + 1}: loss {total_loss / n_batches:.3f}, val acc {val_acc:.3f}")
        # Salvo il migliore sulla validation, non l'ultimo
        if val_acc > best:
            best = val_acc
            save_head(head, "head/head.pt")

    print(f"migliore val acc: {best:.3f}")