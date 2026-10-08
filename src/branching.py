import torch
from transformers import AutoModel, AutoTokenizer
from .prompt import build_branch_prompt

MODEL_NAME = "Qwen/Qwen3-0.6B"
SENTINEL = "<<OPZIONI>>"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device).eval()


def build_inputs(system, before, options, after):
    # Applico il template chat con un segnaposto al posto delle opzioni, poi taglio
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": before + SENTINEL + after},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if text.count(SENTINEL) != 1:
        raise ValueError("Il documento contiene il segnaposto, cambia SENTINEL")
    head, tail = text.split(SENTINEL)
    return head, options, tail


def tokenize_prompt(head, options, tail):
    # Ogni opzione e tokenizzata a parte, cosi so dove inizia e dove finisce
    head_tokens = tokenizer.encode(head)
    tail_tokens = tokenizer.encode(tail)

    final_tokens = list(head_tokens)
    choices_end_idx = []
    for option in options:
        final_tokens.extend(tokenizer.encode(option))
        choices_end_idx.append(len(final_tokens) - 1)
    final_tokens.extend(tail_tokens)

    span = {
        "question_end_idx": len(head_tokens) - 1,
        "choices_end_idx": choices_end_idx,
        "answer_end_idx": len(final_tokens) - 1,
    }
    return final_tokens, span


def get_choice_spans(span):
    # Coppie (inizio, fine) di ogni opzione, estremi inclusi
    spans = []
    start = span["question_end_idx"] + 1
    for end in span["choices_end_idx"]:
        spans.append((start, end))
        start = end + 1
    return spans


def get_attention_mask(span):
    L = span["answer_end_idx"] + 1
    question_len = span["question_end_idx"] + 1
    choice_spans = get_choice_spans(span)
    answer_start = choice_spans[-1][1] + 1
    mask = torch.zeros((L, L), dtype=torch.int)

    # 1. La domanda si guarda solo indietro (causale)
    mask[:question_len, :question_len] = torch.tril(torch.ones((question_len, question_len), dtype=torch.int))

    # 2. Ogni opzione vede la domanda e se stessa (causale), mai le altre opzioni:
    #    e questo che rende i rami indipendenti
    for start, end in choice_spans:
        size = end - start + 1
        mask[start:end + 1, :question_len] = 1
        mask[start:end + 1, start:end + 1] = torch.tril(torch.ones((size, size), dtype=torch.int))

    # 3. La risposta vede tutto quello che sta prima, poi e causale fra i suoi token
    answer_len = L - answer_start
    mask[answer_start:, :answer_start] = 1
    mask[answer_start:, answer_start:] = torch.tril(torch.ones((answer_len, answer_len), dtype=torch.int))
    return mask


def generate_position_ids(final_tokens, span):
    question_len = span["question_end_idx"] + 1
    last_choice_end = span["choices_end_idx"][-1]
    position_ids = []

    # 1. La domanda ha le posizioni normali 0, 1, 2...
    for i in range(question_len):
        position_ids.append(i)

    # 2. Ogni opzione riparte dalla stessa posizione, subito dopo la domanda:
    #    il modello non deve vedere un ordine fra le opzioni
    choice_begin = question_len
    for choice_end in span["choices_end_idx"]:
        for i in range(choice_end - choice_begin + 1):
            position_ids.append(question_len + i)
        choice_begin = choice_end + 1

    # 3. La risposta riprende dall'indice che avrebbe in una sequenza normale
    answer_len = span["answer_end_idx"] - last_choice_end
    for i in range(answer_len):
        position_ids.append(last_choice_end + 1 + i)

    return position_ids


@torch.no_grad()
def get_choice_embeddings(document, categories, descriptions=None):
    system, before, options, after = build_branch_prompt(document, categories, descriptions)
    head, options, tail = build_inputs(system, before, options, after)
    final_tokens, span = tokenize_prompt(head, options, tail)

    input_ids = torch.tensor(final_tokens).unsqueeze(0).to(model.device)
    position_ids = torch.tensor(generate_position_ids(final_tokens, span)).unsqueeze(0).to(model.device)

    # Il modello vuole 0 dove si puo guardare e -inf dove no, con forma (1, 1, L, L)
    mask = get_attention_mask(span).unsqueeze(0).unsqueeze(0)
    mask = torch.where(mask == 0, -float("inf"), torch.tensor(0.0)).to(model.device)

    output = model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids)
    hidden = output.last_hidden_state[0]

    # Un embedding per opzione: quello dell'ultimo token del ramo
    choice_embeddings = []
    for idx in span["choices_end_idx"]:
        choice_embeddings.append(hidden[idx])
    answer_embedding = hidden[-1]

    return torch.stack(choice_embeddings).cpu(), answer_embedding.cpu()
