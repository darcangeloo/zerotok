# zerotok

Formerly RobSort.

Typed decisions from a local LLM by reading logits, without generating text.

## Quickstart (Windows, PowerShell)

```powershell
winget install ggml.llamacpp
.\start_server.ps1
```

In another terminal:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

```powershell
curl -X POST http://127.0.0.1:8000/v1/decisions `
  -H "Content-Type: application/json" `
  -d '{"state":"Il cliente scrive: il pacco e arrivato danneggiato e vuole un rimborso urgente entro oggi.","decisions":[{"name":"urgenza","options":["Alta","Media","Bassa"]},{"name":"sentiment","options":["Negativo","Neutro","Positivo"]}]}'
```

```json
{
  "decisions": [
    {
      "name": "urgenza",
      "choice": "Alta",
      "probs": {"Alta": 0.9999999998, "Media": 0.0000000001, "Bassa": 0.0000000000},
      "latency_ms": 156.3
    },
    {
      "name": "sentiment",
      "choice": "Negativo",
      "probs": {"Negativo": 0.9999999999, "Positivo": 0.0000000000, "Neutro": 0.0000000000},
      "latency_ms": 45.4
    }
  ],
  "total_ms": 201.7
}
```

## How it works

Each decision costs a single forward pass instead of generating one token at a time: the model runs with `max_tokens: 1`, so no text is produced. The prompt is processed in parallel by llama.cpp, and only the logits of the letters allowed as options (A, B, C...) are read and turned into probabilities with softmax. The document is placed before the instructions in the prompt so that repeated decisions on the same document share an identical prefix. Prefix caching itself is a llama.cpp feature; this project's contribution is designing the prompt to exploit it and measuring the effect.

## Results

### Latency (Qwen2.5-1.5B-Instruct)

Same 110-token prompt, median of 20 requests after warm-up:

| Setup | Median latency |
|---|---|
| transformers bf16, in-process, GPU | 45.3 ms |
| llama.cpp, server-side compute | 20.5 ms |
| llama.cpp, end-to-end over HTTP | 36.2 ms |

The 20.5 ms and 36.2 ms figures were measured with Qwen2.5-1.5B-Instruct and an earlier version of `bench_latency.py`; running it again reproduces the same method and gives numbers of the same order, not identical ones. The 45.3 ms transformers figure was measured with a separate script that is not included in this repo. The previous version of this project ran on CPU by mistake (torch without CUDA), so most of the jump from that old version comes from fixing that, not from the runtime change. At equal GPU, the runtime gain is about 2x on compute and about 20% end-to-end.

### Accuracy, zero-shot (Qwen3-4B-Instruct-2507 Q8_0)

- SST-2: 88.0% (100 balanced examples from the validation split)
- AG News: 88.5%, macro F1 0.885 (200 balanced examples from the test split)

AG News errors concentrate on Business vs Sci/Tech.

### Prefix caching (Qwen3-4B), 20 documents x 4 decisions

| Document | First decision | Later decisions |
|---|---|---|
| Long (~500 tokens) | 188.8 ms end-to-end (130.3 ms server) | 44.4 ms end-to-end (23.5 ms server) |
| Short (~80 tokens) | 50.4 ms | 42.3 ms |

The cost of later decisions does not depend on document length: 10 decisions on a 500-token document take about 590 ms instead of about 1.9 s.

### Hardware

Laptop RTX 5060 8GB, i7-14650HX, 16GB RAM.

## Limitations

- Probabilities are overconfident, close to 0 or 1 even on wrong answers; do not use them as calibrated confidence.
- Accuracy drops with overlapping or ambiguous labels: on an internal set of 45 support tickets, accuracy was 51-60% depending on class names and order, with a systematic bias toward high severity.
- The 1.5B model showed a position bias toward option A.
- Single slot (`-np 1`): concurrent clients on different documents invalidate each other's cache.
- About 23 ms server-side is the floor per decision with this model and GPU; about 20 ms is HTTP/Python overhead.
- Contextual calibration was tested and had no effect.

## Reproduce

```powershell
python -m src.prepare_data
python -m src.evaluation data/sst2.json
python -m src.evaluation data/agnews.json
python bench_cache.py
python bench_latency.py
```

`bench_latency.py` compares against the 20.5 ms / 36.2 ms figures above only if the server is started with Qwen2.5-1.5B-Instruct Q8_0 instead of the 4B model.

## Credits

Inspired by Jev (TypeSafe AI) and Rizzo Flow (Rizzo AI Academy).

## License

See [LICENSE](LICENSE).
