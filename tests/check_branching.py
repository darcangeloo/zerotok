"""Self-check per maschera di attenzione e position ids del backend a rami.

Esegui: python -m tests.check_branching
Non scarica il modello: transformers e stubbato prima dell'import.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub di transformers: branching.py carica il modello a import time
class _Fake:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _Fake()

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


sys.modules["transformers"] = types.SimpleNamespace(AutoModel=_Fake, AutoTokenizer=_Fake)

from src import branching as b  # noqa: E402


def demo():
    # Layout: domanda 0-2, opzioni 3-4 / 5-7 / 8, risposta 9-10
    span = {
        "question_end_idx": 2,
        "choices_end_idx": [4, 7, 8],
        "answer_end_idx": 10,
    }

    assert b.get_choice_spans(span) == [(3, 4), (5, 7), (8, 8)]

    mask = b.get_attention_mask(span)
    assert mask.shape == (11, 11)

    # Le opzioni non si vedono fra loro
    assert mask[3:5, 5:9].sum() == 0
    assert mask[5:8, 3:5].sum() == 0 and mask[5:8, 8:9].sum() == 0
    assert mask[8, 3:8].sum() == 0

    # Le opzioni vedono la domanda
    assert mask[3:9, :3].all()

    # La risposta vede tutto cio che la precede
    assert mask[9:11, :9].all()

    # Position ids: opzioni ripartono tutte da 3, risposta riprende da 9
    pos = b.generate_position_ids(list(range(11)), span)
    assert pos == [0, 1, 2, 3, 4, 3, 4, 5, 3, 9, 10], pos

    print("check_branching OK")


if __name__ == "__main__":
    demo()
