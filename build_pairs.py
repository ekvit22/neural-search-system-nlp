import json
import random
from typing import Dict, List

import config
from chunk_book import split_sentences


def ict_pairs(chunk: str, n_per_chunk: int) -> List[List[str]]:
    sentences = split_sentences(chunk)
    if len(sentences) < 2:
        return []
    pairs = []
    for index in random.sample(range(len(sentences)), k=min(n_per_chunk, len(sentences))):
        query = sentences[index]
        positive = " ".join(sentences[:index] + sentences[index + 1:])
        if positive.strip():
            pairs.append([query, positive])
    return pairs


def eval_queries(chunk: str, num_queries: int = 1) -> List[str]:
    sentences = split_sentences(chunk)
    if not sentences:
        return []
    return random.sample(sentences, k=min(num_queries, len(sentences)))


def main() -> None:
    random.seed(config.SEED)
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    indices = list(range(len(chunks)))
    random.shuffle(indices)
    num_train = int(0.8 * len(indices))
    num_val = int(0.1 * len(indices))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    train_pairs = []
    for index in train_indices:
        train_pairs.extend(ict_pairs(chunks[index]["text"], config.PAIRS_PER_CHUNK))

    def build_eval(split_indices: List[int]) -> Dict[str, list]:
        queries, gold = [], []
        for index in split_indices:
            for query in eval_queries(chunks[index]["text"], num_queries=1):
                queries.append(query)
                gold.append(chunks[index]["id"])
        return {"queries": queries, "gold": gold}

    with open(config.TRAIN_PAIRS_PATH, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, ensure_ascii=False)
    with open(config.VAL_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(build_eval(val_indices), f, ensure_ascii=False)
    with open(config.TEST_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(build_eval(test_indices), f, ensure_ascii=False)

    print(f"train pairs: {len(train_pairs)} | val chunks: {len(val_indices)} | test chunks: {len(test_indices)}")
