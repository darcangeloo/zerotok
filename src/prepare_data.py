import json
import random
from datasets import load_dataset

N_PER_CLASS = 50
SEED = 42


def build(hf_name, split, text_col, label_names, out_path):
    ds = load_dataset(hf_name, split=split)
    rng = random.Random(SEED)
    labels = ds["label"]

    examples = []
    for label_id, name in enumerate(label_names):
        indices = [i for i, l in enumerate(labels) if l == label_id]
        for i in rng.sample(indices, N_PER_CLASS):
            examples.append({"text": ds[i][text_col], "label": name})

    rng.shuffle(examples)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"categories": label_names, "examples": examples}, f, ensure_ascii=False, indent=2)
    print(out_path, len(examples), "esempi")


build("stanfordnlp/sst2", "validation", "sentence", ["Negative", "Positive"], "data/sst2.json")
build("fancyzhx/ag_news", "test", "text", ["World", "Sports", "Business", "Sci/Tech"], "data/agnews.json")