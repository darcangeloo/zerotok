import math


def get_letter_probs(top_logprobs, used_letters):
    found = {}
    for item in top_logprobs:
        if item["token"] in used_letters:
            found[item["token"]] = item["logprob"]

    # Se manca una lettera la normalizzazione sarebbe sbagliata, meglio fermarsi
    missing = [l for l in used_letters if l not in found]
    if missing:
        raise ValueError(f"Lettere non presenti nei top_logprobs: {missing}")

    # Softmax solo sulle lettere ammesse, sottraggo il massimo per stabilita numerica
    max_lp = max(found.values())
    exps = {l: math.exp(lp - max_lp) for l, lp in found.items()}
    total = sum(exps.values())
    return {l: e / total for l, e in exps.items()}