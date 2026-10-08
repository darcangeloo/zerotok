import math
import torch
import torch.nn as nn


class ChoiceHead(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        # Le salvo per poterle scrivere nel file dei pesi
        self.hidden_dim = hidden_dim
        self.new_dim = new_dim

        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        # Un solo neurone di output, condiviso da tutte le scelte
        self.score = nn.Linear(new_dim, 1)

    def forward(self, choice_emb, answer_emb, choice_mask):
        # choice_emb: (B, C, H), answer_emb: (B, H), choice_mask: (B, C) True = scelta reale
        x = torch.cat([choice_emb, answer_emb.unsqueeze(1)], dim=1)   # (B, C+1, H)

        # La risposta e sempre valida, il padding riguarda solo le scelte
        answer_valid = torch.ones_like(choice_mask[:, :1])
        valid = torch.cat([choice_mask, answer_valid], dim=1)         # (B, C+1)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        att = q @ k.transpose(1, 2) / math.sqrt(self.new_dim)         # (B, C+1, C+1)
        # Nessuno deve guardare il padding
        att = att.masked_fill(~valid.unsqueeze(1), float("-inf"))
        weights = torch.softmax(att, dim=-1)

        # Residuo: ogni scelta mantiene il proprio contenuto
        out = weights @ v + v

        # Un punteggio per scelta, la posizione della risposta la scarto
        logits = self.score(out[:, :-1]).squeeze(-1)                  # (B, C)
        # Il padding non puo vincere
        return logits.masked_fill(~choice_mask, float("-inf"))


def save_head(head, path, temperature=1.0):
    torch.save(
        {
            "hidden_dim": head.hidden_dim,
            "new_dim": head.new_dim,
            "state_dict": head.state_dict(),
            "temperature": temperature,
        },
        path,
    )


def load_head(path):
    checkpoint = torch.load(path, map_location="cpu")
    head = ChoiceHead(checkpoint["hidden_dim"], checkpoint["new_dim"])
    head.load_state_dict(checkpoint["state_dict"])
    # I checkpoint vecchi non hanno la temperatura: 1.0 non cambia nulla
    return head.eval(), checkpoint.get("temperature", 1.0)