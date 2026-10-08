import sys
from pathlib import Path

# Permette di lanciare lo script anche direttamente: aggiungo la root al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from collections import Counter
from scripts.train_head import build_sets

random.seed(0)

# Stesso seed e stessa divisione per stato: gli esempi sono gli stessi del training
train_examples, val_examples = build_sets()

correct = Counter()  # quante volte un'opzione era la risposta giusta
seen = Counter()     # quante volte e comparsa
for _, choices, label in train_examples:
    for c in choices:
        seen[c] += 1
    correct[choices[label]] += 1


def prior(choice):
    return correct[choice] / seen[choice] if seen[choice] else 0.0


hits = 0
for _, choices, label in val_examples:
    scores = [prior(c) for c in choices]
    best = max(scores)
    # A parita scelgo a caso, altrimenti favorirei la prima opzione
    pred = random.choice([i for i, s in enumerate(scores) if s == best])
    hits += pred == label

print(f"baseline a priori: {hits / len(val_examples):.3f}")