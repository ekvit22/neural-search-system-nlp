import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from build_tokenizer import load_tokenizer
from data import TextDataset, text_collate
from model.model import TransformerEncoderModel


class VectorSearch:
    def __init__(self, model, tokenize, pad_id, chunks, device):
        self.model = model
        self.tokenize = tokenize
        self.pad_id = pad_id
        self.chunks = chunks  # list of {"id": int, "text": str}
        self.device = device
        self.index = None  # (num_chunks, d_model) normalized tensor

    def build_index(self, batch_size=64):
        texts = [c["text"] for c in self.chunks]
        dataset = TextDataset(texts, self.tokenize)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda b: text_collate(b, self.pad_id),
        )
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                emb = self.model(input_ids.to(self.device), attention_mask.to(self.device))
                embeddings.append(emb.cpu())
        self.index = F.normalize(torch.cat(embeddings, dim=0), dim=-1)
        print(f"index built: {self.index.shape[0]} chunks x {self.index.shape[1]} dims")

    def retrieve(self, query: str, k: int) -> list:
        assert self.index is not None, "call build_index() first"
        self.model.eval()
        with torch.no_grad():
            ids = torch.tensor(
                [self.tokenize(query)[: config.MAX_LEN]], dtype=torch.long
            ).to(self.device)
            mask = torch.ones_like(ids)
            q_emb = F.normalize(self.model(ids, mask), dim=-1).cpu()
        scores = (q_emb @ self.index.T)[0]
        top_indices = scores.topk(k).indices.tolist()
        return [self.chunks[i]["id"] for i in top_indices]


def load(
    model_path=config.MODEL_PATH,
    tokenizer_path=config.TOKENIZER_PATH,
    chunks_path=config.CHUNKS_PATH,
):
    tokenize, pad_id, vocab_size = load_tokenizer(tokenizer_path)
    device = torch.device(config.DEVICE)

    model = TransformerEncoderModel(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FF,
        max_len=config.MODEL_MAX_LEN,
        dropout=config.DROPOUT,
        pad_id=pad_id,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    vs = VectorSearch(model, tokenize, pad_id, chunks, device)
    vs.build_index()
    return vs
