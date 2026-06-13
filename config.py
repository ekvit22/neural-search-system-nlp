import os
import torch


def _int(name, default):
    return int(os.environ.get(name, default))


def _flt(name, default):
    return float(os.environ.get(name, default))


SEED = _int("NS_SEED", 42)

# paths
DATA_DIR = os.environ.get("NS_DATA_DIR", "data")
# MS MARCO training/eval data
PASSAGES_PATH = os.path.join(DATA_DIR, "passages.json")
QUERIES_META_PATH = os.path.join(DATA_DIR, "queries_meta.json")
TRAIN_PAIRS_PATH = os.path.join(DATA_DIR, "train_pairs.json")
VAL_EVAL_PATH = os.path.join(DATA_DIR, "val_eval.json")
TEST_EVAL_PATH = os.path.join(DATA_DIR, "test_eval.json")
# book demo data
BOOK_PATH = os.path.join(DATA_DIR, "book.txt")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
TOKENIZER_PATH = os.environ.get("NS_TOKENIZER", "tokenizer.json")
MODEL_PATH = os.environ.get("NS_MODEL", "best_model.pt")

# chunking
CHUNK_MIN = _int("NS_CHUNK_MIN", 200)
CHUNK_MAX = _int("NS_CHUNK_MAX", 300)

# tokenizer
VOCAB_SIZE = _int("NS_VOCAB_SIZE", 8000)

# model
D_MODEL = _int("NS_D_MODEL", 256)
NHEAD = _int("NS_NHEAD", 4)
NUM_LAYERS = _int("NS_NUM_LAYERS", 4)
DIM_FF = _int("NS_DIM_FF", 512)
DROPOUT = _flt("NS_DROPOUT", 0.1)
MAX_LEN = _int("NS_MAX_LEN", 128)
MODEL_MAX_LEN = _int("NS_MODEL_MAX_LEN", 512)

# training
FINETUNE_EPOCHS = _int("NS_FINETUNE_EPOCHS", 15)
FINETUNE_BS = _int("NS_FINETUNE_BS", 128)
FINETUNE_LR = _flt("NS_FINETUNE_LR", 2e-4)
TEMPERATURE = _flt("NS_TEMPERATURE", 0.05)

# encoding
ENCODE_BS = _int("NS_ENCODE_BS", 512)

# eval
EVAL_K = _int("NS_EVAL_K", 10)
PAIRS_PER_CHUNK = _int("NS_PAIRS_PER_CHUNK", 3)

DEVICE = os.environ.get("NS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")