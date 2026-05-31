import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()         
    summed = (hidden * mask).sum(dim=1)                   
    counts = mask.sum(dim=1).clamp(min=1e-9)              
    return summed / counts
 
 
class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))       
 
    def forward(self, x):                                 
        return x + self.pe[:, : x.size(1)]


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

class TransformerBlock(nn.Module): 
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, key_padding_mask):
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
 
 
class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=512, max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len)
        self.in_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
 
    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        x = self.in_dropout(self.pos(x))
        key_padding_mask = attention_mask == 0              
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.final_norm(x)
        return masked_mean(x, attention_mask)         

class BagOfEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model=256, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, input_ids, attention_mask):
        x = self.dropout(self.embed(input_ids))
        return masked_mean(x, attention_mask)
