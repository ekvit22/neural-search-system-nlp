"""Split the Jurafsky & Martin book into 200-300 word passages for the search corpus.

The model is trained on MS MARCO, but the demo/search index must be the book
(per the assignment). This turns book.txt into book_chunks.json, a list of
{"id": int, "text": str} objects — the same format vector_search expects.

Usage:
    python chunk_book.py [book.txt] [book_chunks.json] [min_words] [max_words]
"""
import json
import re
import sys


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, min_words: int = 200, max_words: int = 300) -> list:
    """Greedily pack whole sentences into chunks, never cutting mid-sentence."""
    chunks, current, count = [], [], 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if count + words > max_words and count >= min_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current and count >= min_words // 4:   # keep a short tail chunk
        chunks.append(" ".join(current))
    return chunks


def main(book_path: str = "book.txt", out_path: str = "book_chunks.json",
         min_words: int = 200, max_words: int = 300) -> None:
    with open(book_path, encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, int(min_words), int(max_words))
    objects = [{"id": i, "text": c} for i, c in enumerate(chunks)]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(objects, f, ensure_ascii=False)

    counts = [len(c.split()) for c in chunks]
    print(f"chunked into {len(chunks)} passages -> {out_path}")
    if counts:
        print(f"word counts: min={min(counts)} mean={sum(counts) // len(counts)} max={max(counts)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
