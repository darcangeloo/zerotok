import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_letter_probs(logits, letters, tokenizer):
    letter_ids = []
    
    for letter in letters:
        ids = tokenizer.encode(letter)
        letter_ids.append(ids[0])
        
    letter_logits = logits[letter_ids]
    probs = torch.softmax(letter_logits, dim=0).tolist()
    probas_dict = dict(zip(letters, probs))
    return probas_dict
