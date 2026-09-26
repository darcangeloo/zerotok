import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.classify import classify_document

app = FastAPI(title="zerotok")

# Il server llama.cpp gira con un solo slot (-np 1): senza questo lock,
# due richieste concorrenti intreccerebbero le loro decisioni e invaliderebbero la cache a vicenda
lock = threading.Lock()


class Decision(BaseModel):
    name: str
    options: list[str]
    descriptions: list[str] | None = None


class DecisionsRequest(BaseModel):
    state: str
    decisions: list[Decision]


@app.post("/v1/decisions")
def decisions(request: DecisionsRequest):
    for d in request.decisions:
        if len(d.options) < 2 or len(d.options) > 26:
            raise HTTPException(400, f"'{d.name}': options deve avere tra 2 e 26 elementi")
        if d.descriptions is not None and len(d.descriptions) != len(d.options):
            raise HTTPException(400, f"'{d.name}': descriptions deve avere la stessa lunghezza di options")

    total_start = time.perf_counter()
    results = []
    with lock:
        for d in request.decisions:
            start = time.perf_counter()
            probs = classify_document(request.state, d.options, d.descriptions)
            latency_ms = (time.perf_counter() - start) * 1000

            letter = max(probs, key=probs.get)
            choice = d.options[ord(letter) - ord("A")]
            named_probs = {d.options[ord(l) - ord("A")]: p for l, p in probs.items()}

            results.append({
                "name": d.name,
                "choice": choice,
                "probs": named_probs,
                "latency_ms": latency_ms,
            })
    total_ms = (time.perf_counter() - total_start) * 1000

    return {"decisions": results, "total_ms": total_ms}


@app.get("/")
def index():
    return FileResponse("index.html")
