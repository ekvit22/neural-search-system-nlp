import torch
import torch.nn.functional as F


def info_nce(query_emb, passage_emb, temperature=0.05):
    q = F.normalize(query_emb, dim=-1)
    p = F.normalize(passage_emb, dim=-1)
    logits = (q @ p.t()) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss_q = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_q + loss_p)


def triplet_nce(query_emb, pos_emb, neg_emb, temperature: float = 0.05):
    q   = F.normalize(query_emb, dim=-1)
    pos = F.normalize(pos_emb, dim=-1)
    neg = F.normalize(neg_emb, dim=-1)
    all_p  = torch.cat([pos, neg], dim=0)
    logits = (q @ all_p.t()) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)
