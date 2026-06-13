import json
import os
import tempfile
from typing import Callable, List, Tuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import config

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _text_iterator():
    if hasattr(config, "QUERIES_META_PATH") and os.path.exists(config.QUERIES_META_PATH):
        with open(config.QUERIES_META_PATH, encoding="utf-8") as f:
            for item in json.load(f):
                yield item["query"]
    if hasattr(config, "PASSAGES_PATH") and os.path.exists(config.PASSAGES_PATH):
        with open(config.PASSAGES_PATH, encoding="utf-8") as f:
            for item in json.load(f):
                yield item["text"]
    elif hasattr(config, "CORPUS_PATH") and os.path.exists(config.CORPUS_PATH):
        with open(config.CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def train_tokenizer(
    vocab_size: int = config.VOCAB_SIZE,
    out_path: str = config.TOKENIZER_PATH,
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        for text in _text_iterator():
            tmp.write(text.replace("\n", " ") + "\n")
        tmp_path = tmp.name

    tokenizer.train([tmp_path], trainer)
    os.unlink(tmp_path)
    tokenizer.save(out_path)
    print(f"saved tokenizer -> {out_path} (vocab={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(
    path: str = config.TOKENIZER_PATH,
) -> Tuple[Callable[[str], List[int]], int, int]:
    tokenizer = Tokenizer.from_file(path)
    pad_id = tokenizer.token_to_id("[PAD]")
    vocab_size = tokenizer.get_vocab_size()
    tokenize = lambda text: tokenizer.encode(text).ids
    return tokenize, pad_id, vocab_size


def main() -> None:
    train_tokenizer()
