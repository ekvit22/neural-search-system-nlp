import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, key_padding_mask):
        B, L, d = x.shape
        def split(t):
            return t.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
 
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if key_padding_mask is not None:
            mask = key_padding_mask.view(B, 1, 1, L)                  
            scores = scores.masked_fill(mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        ctx = attention @ v                                                
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, d)          
        return self.out_proj(ctx)