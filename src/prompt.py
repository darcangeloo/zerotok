categories = ["Fattura", "Contratto", "Altro"]

def build_prompt(document, categories):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = []
    for (idx, category) in enumerate(categories):
        letter = letters[idx]
        single_row = f"{letter}. {category}"
        rows.append(single_row)
        
    used_letters = list(letters[:len(categories)])
    options_text = "\n".join(rows)
    letters_text = ", ".join(used_letters)
    
    prompt = f"""Sei un classificatore di documenti aziendali.

Documento:
{document}

Classifica il documento in UNA SOLA delle seguenti categorie:

{options_text}

IMPORTANTE:
- Scegli "Altro" se il documento non riguarda una fattura, un contratto o le altre categorie specificate.
- Non scegliere una categoria solo perché il documento contiene parole vagamente correlate.
- Devi classificare in base allo scopo principale del documento.

Rispondi con una sola lettera maiuscola ({letters_text}). Non aggiungere altro testo."""

    return prompt, used_letters
