import json
import os
import re
from typing import List

import config


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, min_words: int = config.CHUNK_MIN, max_words: int = config.CHUNK_MAX) -> List[str]:
    chunks, current, current_words = [], [], 0
    for sentence in split_sentences(text):
        word_count = len(sentence.split())
        if current_words + word_count > max_words and current_words >= min_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += word_count
    if current and current_words >= max(1, min_words // 4):
        chunks.append(" ".join(current))
    return chunks


def main() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.BOOK_PATH, encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    chunk_objects = [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_objects, f, ensure_ascii=False)
    print(f"chunked into {len(chunks)} passages -> {config.CHUNKS_PATH}")
