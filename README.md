# RobSort

**Local-first classification for business documents.**

RobSort classifies a document against categories supplied at runtime—such as `Fattura`, `Contratto`, and `Altro`—using next-token probabilities from the local **Qwen2.5-1.5B-Instruct** model. Documents do not need to be sent to an external API.

> Your documents stay on your machine.

## Features

- Local inference, with CUDA used when it is available.
- Configurable categories for every request.
- FastAPI endpoint returning category, confidence, and probabilities.
- Browser dashboard written in vanilla HTML, CSS, and JavaScript.
- A small built-in evaluation set with accuracy, recall, F1, timing, and a confusion matrix.

## Architecture

```text
Document → prompt with category labels → local Qwen model → logits → softmax → predicted category
```

## Project structure

```text
RobSort/
├── app.py                 # FastAPI application and dashboard route
├── evaluation.py          # Evaluation data and metrics
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── src/
    ├── classify.py        # Model loading and inference
    ├── logits.py          # Logits → probabilities
    └── prompt.py          # Classification prompt
```

## Installation

Use Python 3.10+. Create a virtual environment and install the required packages.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]" transformers accelerate bitsandbytes torch
```

On its first run, RobSort downloads `Qwen/Qwen2.5-1.5B-Instruct`. The current model configuration uses four-bit loading and is intended for a compatible CUDA environment. For CPU-only use, update the model-loading options in `src/classify.py`.

## Run locally

```powershell
uvicorn app:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The frontend and API are served from the same local application, so CORS configuration is not required.

## API

`POST /classify`

```json
{
  "document": "Fattura n. 482/2026. Imponibile €1.250, IVA 22%, totale €1.525.",
  "categories": ["Fattura", "Contratto", "Altro"]
}
```

```json
{
  "category": "A",
  "confidence": 0.8818,
  "probabilities": {
    "A": 0.8818,
    "B": 0.1175,
    "C": 0.0008
  }
}
```

The letter labels correspond to the supplied category order: `A → Fattura`, `B → Contratto`, `C → Altro`. `evaluation.py` maps those letters to readable names before computing its metrics.

## How it works

1. `src/prompt.py` labels the supplied categories as `A`, `B`, `C`, and creates a constrained prompt.
2. `src/classify.py` sends the prompt to the local Qwen model.
3. `src/logits.py` selects the logits for permitted output letters and applies softmax.
4. The highest probability becomes the predicted class.

The confidence is normalized across the supplied category options; it should not be interpreted as a calibrated probability without further calibration work.

## Next steps

- Expand the evaluation set with anonymized, held-out real documents.
- Map probability keys to category names in the API response.
- Add PDF/DOCX text extraction before classification.
- Add automated tests and confidence calibration.

## License

See [LICENSE](LICENSE).
