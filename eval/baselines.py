

import re
from typing import List

from rank_bm25 import BM25Okapi 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Search:

    def __init__(self, passages):
        self.passages = passages
        corpus = [_simple_tokenize(p["text"]) for p in passages]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, k: int) -> List[int]:
        scores = self.bm25.get_scores(_simple_tokenize(query))
        top = scores.argsort()[::-1][:k]
        return [self.passages[i]["id"] for i in top]


class TfidfSearch:

    def __init__(self, passages):
        self.passages = passages
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.doc_matrix = self.vectorizer.fit_transform(p["text"] for p in passages)

    def retrieve(self, query: str, k: int) -> List[int]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [self.passages[i]["id"] for i in top]
