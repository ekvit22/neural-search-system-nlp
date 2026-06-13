
import sys

import config

try:
    from eval.evaluate import NeuralRetriever, load_passages
except ImportError:
    from evaluate import NeuralRetriever, load_passages


def make_search():
    passages = load_passages()
    text_lookup = {p["id"]: p["text"] for p in passages}
    retriever = NeuralRetriever(passages).build_index()

    def search(query: str, k: int = 5, preview: int = 280):
        print(f"\nQuery: {query}\n" + "=" * 70)
        for rank, pid in enumerate(retriever.retrieve(query, k), start=1):
            snippet = text_lookup[pid][:preview].replace("\n", " ")
            print(f"[{rank}] passage {pid}: {snippet}...\n")

    return search


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "what is a transformer and how does attention work"
    search = make_search()
    search(query)
