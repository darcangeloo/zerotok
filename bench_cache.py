import json
import statistics
import time

from src.classify import request_logprobs
from src.prompt import build_prompt

# Decisioni diverse sullo stesso documento. Qui misuriamo solo la latenza,
# l'accuracy di queste decisioni extra non viene valutata.
DECISIONS = [
    ["World", "Sports", "Business", "Sci/Tech"],
    ["Negative", "Neutral", "Positive"],
    ["Europe", "Americas", "Asia", "Africa/Middle East", "Global"],
    ["Breaking news", "Analysis", "Announcement"],
]
N_DOCS = 21          # il primo è warm-up e viene scartato
LONG_JOIN = 10       # quante notizie unire per il documento lungo


def decide(document, categories):
    messages, _ = build_prompt(document, categories)
    start = time.perf_counter()
    data = request_logprobs(messages)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, data.get("timings", {})


def run(docs, label):
    first, rest, server_first, server_rest = [], [], [], []
    for i, doc in enumerate(docs):
        for j, cats in enumerate(DECISIONS):
            elapsed, t = decide(doc, cats)
            if i == 0:
                if j == 0:
                    print("timings di esempio:", t)
                continue
            (first if j == 0 else rest).append(elapsed)
            if "prompt_ms" in t:
                (server_first if j == 0 else server_rest).append(t["prompt_ms"])

    print(f"\n{label}")
    print(f"  prima decisione      {statistics.median(first):.1f} ms end-to-end")
    print(f"  decisioni successive {statistics.median(rest):.1f} ms end-to-end")
    if server_first:
        print(f"  lato server: prima {statistics.median(server_first):.1f} ms, successive {statistics.median(server_rest):.1f} ms")


with open("data/agnews.json", encoding="utf-8") as f:
    texts = [ex["text"] for ex in json.load(f)["examples"]]

short_docs = texts[:N_DOCS]
long_docs = ["\n\n".join(texts[k * LONG_JOIN:(k + 1) * LONG_JOIN]) for k in range(N_DOCS)]

run(short_docs, "documenti corti (1 notizia)")
run(long_docs, f"documenti lunghi ({LONG_JOIN} notizie)")