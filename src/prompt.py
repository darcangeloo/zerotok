LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYSTEM = "Sei un sistema che prende decisioni. Rispondi solo con la lettera dell'opzione scelta."


def build_prompt(document, categories, descriptions=None):
    if len(categories) > len(LETTERS):
        raise ValueError("Troppe categorie, massimo 26")
    if descriptions is not None and len(descriptions) != len(categories):
        raise ValueError("descriptions deve avere la stessa lunghezza di categories")

    used_letters = list(LETTERS[:len(categories)])
    if descriptions:
        options = "\n".join(f"{letter}) {cat}: {desc}" for letter, cat, desc in zip(used_letters, categories, descriptions))
    else:
        options = "\n".join(f"{letter}) {cat}" for letter, cat in zip(used_letters, categories))

    # Il documento va prima delle istruzioni, cosi resta un prefisso identico
    # e il server puo riusare la cache quando si fanno piu decisioni sullo stesso documento
    user = (
        f"Documento:\n{document}\n\n"
        f"Scegli la categoria del documento tra queste opzioni:\n{options}\n\n"
        f"Rispondi solo con la lettera."
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    return messages, used_letters