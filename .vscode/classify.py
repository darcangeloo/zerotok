from src.logits import get_letter_probs
from src.prompt import build_prompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", load_in_8bit=True, device_map="auto")


    
def classify(document, categories):
    prompt_raw, used_letters = build_prompt(document, categories)
    
    messages = [
        {"role": "user", "content": prompt_raw}
    ]
    
    chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_template, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits[0, -1, :]
    return get_letter_probs(logits, used_letters, tokenizer)

