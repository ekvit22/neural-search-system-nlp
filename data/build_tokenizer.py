from typing import Callable, List, Tuple

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

import config

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_tokenizer(files: List[str], vocab_size: int = config.VOCAB_SIZE, out_path: str = config.TOKENIZER_PATH) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train(files, trainer)
    tokenizer.save(out_path)
    print(f"saved tokenizer -> {out_path}  (vocab={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str = config.TOKENIZER_PATH) -> Tuple[Callable[[str], List[int]], int, int]:
    tokenizer = Tokenizer.from_file(path)
    pad_id = tokenizer.token_to_id("[PAD]")
    vocab_size = tokenizer.get_vocab_size()
    tokenize = lambda text: tokenizer.encode(text).ids
    return tokenize, pad_id, vocab_size


def main() -> None:
    train_tokenizer([config.BOOK_PATH])
