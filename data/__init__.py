from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

import config


class PairDataset(Dataset):

    def __init__(self, pairs, tokenize: Callable[[str], List[int]], max_length: int = config.MAX_LEN) -> None:
        self.pairs = pairs
        self.tokenize = tokenize
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        query, passage = self.pairs[index]
        return self.tokenize(query)[: self.max_length], self.tokenize(passage)[: self.max_length]


class TripletDataset(Dataset):

    def __init__(self, triplets, tokenize: Callable[[str], List[int]], max_length: int = config.MAX_LEN) -> None:
        self.triplets = triplets
        self.tokenize = tokenize
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int], List[int]]:
        query, positive, negative = self.triplets[index]
        return (
            self.tokenize(query)[: self.max_length],
            self.tokenize(positive)[: self.max_length],
            self.tokenize(negative)[: self.max_length],
        )


class TextDataset(Dataset):

    def __init__(self, texts, tokenize: Callable[[str], List[int]], max_length: int = config.MAX_LEN) -> None:
        self.texts = texts
        self.tokenize = tokenize
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> List[int]:
        return self.tokenize(self.texts[index])[: self.max_length]


def pad_batch(sequences: List[List[int]], pad_id: int = 0) -> Tuple[Tensor, Tensor]:
    sequences = [seq if len(seq) > 0 else [pad_id] for seq in sequences]
    max_length = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask


def pair_collate(batch, pad_id: int = 0) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    queries, passages = zip(*batch)
    return pad_batch(queries, pad_id), pad_batch(passages, pad_id)


def triplet_collate(batch, pad_id: int = 0):
    queries, positives, negatives = zip(*batch)
    return pad_batch(queries, pad_id), pad_batch(positives, pad_id), pad_batch(negatives, pad_id)


def text_collate(batch, pad_id: int = 0) -> Tuple[Tensor, Tensor]:
    return pad_batch(batch, pad_id)
