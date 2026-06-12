# neural-search-system-nlp

A neural semantic search system. A Transformer encoder (trained from scratch) maps queries
and passages into a shared embedding space; retrieval is nearest-neighbour by cosine
similarity. Trained contrastively on MS MARCO (v2.1) query–passage triplets.

- **Person 1:** model architecture (`model/`) + InfoNCE
- **Person 2:** data pipeline (MS MARCO → `passages.json`, `train_pairs.json`, eval sets) + `eval/metrics.py`
- **Person 3:** training loop (`train/trainer.py`) + vector search (`eval/vector_search.py`) + `training.ipynb`
- **Person 4:** baselines (TF-IDF / BM25), evaluator, demo

---

## Training Experiments (Person 3)

Loss: `triplet_nce` — in-batch contrastive loss where each query is scored against
`2 × batch_size` candidates (the batch's positives + one hard negative per example).
Random-guess cross-entropy for batch size 64 is `ln(128) ≈ 4.85`.

### Run 1 — baseline

| Setting | Value |
|---|---|
| Train triples | 24,454 (MS MARCO, `MAX_EXAMPLES=50,000`) |
| Val triples | 3,056 |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 1e-4 (AdamW, warmup + cosine decay) |
| Max length | 128 tokens |
| Temperature | 0.05 |
| Model | 4-layer Transformer, d_model=256, vocab=8,000 |

**Validation loss per epoch:**

| Epoch | val_loss | Δ |
|---|---|---|
| 1 | 2.4808 | — |
| 2 | 2.1881 | −0.293 |
| 3 | 2.0723 | −0.116 |
| 4 | 2.0072 | −0.065 |
| 5 | 1.9657 | −0.042 |
| 6 | 1.9441 | −0.022 |
| 7 | 1.9302 | −0.014 |
| 8 | 1.9242 | −0.006 |
| 9 | 1.9222 | −0.002 |
| 10 | 1.9217 | −0.0005 |

**Analysis:**

- **Optimization converged cleanly.** Smooth monotonic descent, no instability, no val-loss
  bounce → no overfitting. The training loop itself is working correctly.
- **The model learned a real signal.** Final loss 1.92 vs. a random baseline of 4.85; the model
  places ~`e^(-1.92) ≈ 15%` of probability mass on the correct passage among 128 candidates
  (vs. 0.8% at chance) — roughly 18× better than random.
- **It is data/capacity-limited, not under-trained.** The curve is flat by epoch 8 (epoch 9→10
  improves by 0.0005). More epochs would not help; the cosine LR has also annealed to ~0.
- **Caveat — loss flatters the model.** Validation loss measures an easy proxy: 1-of-128 in-batch.
  Real retrieval is 1-of-488,335 over the full corpus, so true Recall@K / MRR will be lower.
  Final quality must be judged by the retrieval metrics (Person 4), not this loss.

**Limiting factors identified:** (1) only 24,454 training triples — 3% of MS MARCO's 808k train
split; (2) batch size 64 gives only 127 in-batch negatives per query.

### Run 2 — more data + larger batch + higher LR

Two changes target the two bottlenecks identified in Run 1:

1. **More data** (the high-leverage change) — `MAX_EXAMPLES` in `data-pipeline.ipynb` raised
   50k → 200k MS MARCO examples (~4× more training triples). Requires regenerating the dataset
   and re-uploading it.
2. **Larger batch + scaled LR** — more in-batch negatives per query (the main driver of
   contrastive retrieval quality); LR scaled up for the larger batch, epochs raised to keep the
   gradient-update count comparable.

| Setting | Run 1 | Run 2 |
|---|---|---|
| Train triples | 24,454 | **~98,000** (200k examples) |
| Batch size | 64 | **128** |
| Learning rate | 1e-4 | **2e-4** |
| Epochs | 10 | **15** |
| Negatives / query | 127 | **255** |

> Note: the passage corpus also grows (~488k → ~1.8M), so absolute Recall@K will drop for
> *every* retriever (more distractors). The assignment-relevant signal is the **gap vs. BM25**
> on the same corpus, not the absolute number.

_Results: pending re-run._
