import statistics
import time

from src.classify import request_logprobs
from src.prompt import build_prompt

DOCUMENT = "Il prodotto che ho ricevuto è arrivato rotto e non risponde nessuno al telefono da tre giorni, sono molto arrabbiato."
CATEGORIES = ["Grave", "Neutro", "Non grave"]
N = 21  # la prima è warm-up e viene scartata

messages, _ = build_prompt(DOCUMENT, CATEGORIES)

client_ms, server_ms = [], []
for i in range(N):
    start = time.perf_counter()
    data = request_logprobs(messages)
    elapsed = (time.perf_counter() - start) * 1000
    if i == 0:
        continue  # warm-up scartato
    client_ms.append(elapsed)
    server_ms.append(data.get("timings", {}).get("prompt_ms", float("nan")))

print(f"mediana end-to-end: {statistics.median(client_ms):.1f} ms")
print(f"mediana lato server (prompt_ms): {statistics.median(server_ms):.1f} ms")
