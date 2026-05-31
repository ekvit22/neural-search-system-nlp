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
    