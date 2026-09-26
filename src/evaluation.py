import json
import time
import statistics
from .classify import classify_document
import sys

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/eval.json"

with open(DATASET, encoding="utf-8") as f:
    data = json.load(f)

categories = data["categories"]
examples = data["examples"]

y_true = []
y_pred = []
times = []

for ex in examples:
    start = time.perf_counter()
    probs = classify_document(ex["text"], categories)
    times.append((time.perf_counter() - start) * 1000)

    letter = max(probs, key=probs.get)
    y_true.append(ex["label"])
    y_pred.append(categories[ord(letter) - ord("A")])

# Accuracy
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
print(f"accuracy: {correct / len(examples):.3f} ({correct}/{len(examples)})")

# Precision, recall e F1 per ogni categoria
print("\ncategoria | precision | recall | f1 | esempi")
f1_list = []
for cat in categories:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p == cat)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cat and p == cat)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p != cat)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    f1_list.append(f1)

    print(f"{cat} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {tp + fn}")

# Media semplice delle F1, pesa tutte le categorie allo stesso modo
print(f"\nmacro F1: {sum(f1_list) / len(f1_list):.3f}")

# Matrice di confusione, righe = vero, colonne = predetto
print("\nmatrice di confusione (righe vero, colonne predetto)")
print("\t" + "\t".join(categories))
for true_cat in categories:
    row = [sum(1 for t, p in zip(y_true, y_pred) if t == true_cat and p == pred_cat) for pred_cat in categories]
    print(true_cat + "\t" + "\t".join(str(n) for n in row))

# Latenza
print(f"\nlatenza mediana: {statistics.median(times):.1f} ms")
print(f"latenza massima: {max(times):.1f} ms")