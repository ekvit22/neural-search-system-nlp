
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config

try:
    from data.build_tokenizer import load_tokenizer
except ImportError:
    from build_tokenizer import load_tokenizer

try:
    from model.model import TransformerEncoderModel
except ImportError:
    from model import TransformerEncoderModel

try:
    from eval.metrics import macro_recall_at_k, mean_reciprocal_rank
except ImportError:
    from metrics import macro_recall_at_k, mean_reciprocal_rank

try:
    from eval.baselines import BM25Search, TfidfSearch
except ImportError:
    from baselines import BM25Search, TfidfSearch

from data import TextDataset, text_collate


def load_eval_set(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"], data["gold"]


def _default_passages_path():
    for attr in ("PASSAGES_PATH", "CHUNKS_PATH"):
        p = getattr(config, attr, None)
        if p:
            return p
    return "passages.json"


def load_passages(path=None):
    path = path or _default_passages_path()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class NeuralRetriever:


    def __init__(self, passages):
        self.passages = passages
        self.tokenize, self.pad_id, vocab_size = load_tokenizer()
        self.device = torch.device(config.DEVICE)
        self.model = TransformerEncoderModel(
            vocab_size=vocab_size,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            dim_feedforward=config.DIM_FF,
            max_len=config.MODEL_MAX_LEN,
            dropout=config.DROPOUT,
            pad_id=self.pad_id,
        ).to(self.device)
        self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
        self.model.eval()
        self.index = None

    def build_index(self, batch_size=None):
        batch_size = batch_size or getattr(config, "ENCODE_BS", 256)
        texts = [p["text"] for p in self.passages]
        loader = DataLoader(
            TextDataset(texts, self.tokenize),
            batch_size=batch_size,
            collate_fn=lambda b: text_collate(b, self.pad_id),
        )
        embeddings = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                emb = self.model(input_ids.to(self.device), attention_mask.to(self.device))
                embeddings.append(emb.cpu())
        self.index = F.normalize(torch.cat(embeddings, dim=0), dim=-1)
        print(f"neural index: {self.index.shape[0]:,} passages x {self.index.shape[1]} dims")
        return self

    def retrieve(self, query: str, k: int) -> list:
        assert self.index is not None, "call build_index() first"
        with torch.no_grad():
            ids = torch.tensor(
                [self.tokenize(query)[: config.MAX_LEN]], dtype=torch.long
            ).to(self.device)
            mask = torch.ones_like(ids)
            q_emb = F.normalize(self.model(ids, mask), dim=-1).cpu()
        top = (q_emb @ self.index.T)[0].topk(k).indices.tolist()
        return [self.passages[i]["id"] for i in top]


def evaluate_retriever(retriever, queries, gold, ks=(1, 5, 10)):
    max_k = max(ks)
    retrieved = [retriever.retrieve(q, max_k) for q in queries]
    results = {f"Recall@{k}": macro_recall_at_k(retrieved, gold, k) for k in ks}
    results["MRR"] = mean_reciprocal_rank(retrieved, gold)
    return results


def print_table(scores_by_name, ks=(1, 5, 10)):
    cols = [f"Recall@{k}" for k in ks] + ["MRR"]
    header = f"{'Retriever':<16}" + "".join(f"{c:>11}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, scores in scores_by_name.items():
        row = f"{name:<16}" + "".join(f"{scores[c]:>11.4f}" for c in cols)
        print(row)


def main(eval_path=config.TEST_EVAL_PATH, save_path="eval_results.json", max_queries=None):
    passages = load_passages()
    queries, gold = load_eval_set(eval_path)
    if max_queries: 
        queries, gold = queries[:max_queries], gold[:max_queries]
    print(f"corpus: {len(passages):,} passages | eval queries: {len(queries):,}\n")

    retrievers = {
        "BM25": BM25Search(passages),
        "TF-IDF": TfidfSearch(passages),
        "Neural (ours)": NeuralRetriever(passages).build_index(),
    }

    scores_by_name = {
        name: evaluate_retriever(r, queries, gold) for name, r in retrievers.items()
    }

    print()
    print_table(scores_by_name)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(scores_by_name, f, indent=2)
        print(f"\nsaved -> {save_path}")

    return scores_by_name


if __name__ == "__main__":
    main()
