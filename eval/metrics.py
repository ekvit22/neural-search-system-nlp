from typing import List


def recall_at_k(retrieved_ids: List[int], relevant_id: int, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids: List[int], relevant_id: int) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid == relevant_id:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(retrieved_list: List[List[int]], gold_ids: List[int]) -> float:
    scores = [reciprocal_rank(ret, gold) for ret, gold in zip(retrieved_list, gold_ids)]
    return sum(scores) / len(scores) if scores else 0.0


def macro_recall_at_k(retrieved_list: List[List[int]], gold_ids: List[int], k: int) -> float:
    scores = [recall_at_k(ret, gold, k) for ret, gold in zip(retrieved_list, gold_ids)]
    return sum(scores) / len(scores) if scores else 0.0
