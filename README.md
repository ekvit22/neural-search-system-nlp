# neural-search-system-nlp

A neural semantic search system. A Transformer encoder, trained from scratch, maps queries and
passages into a shared embedding space; retrieval is nearest-neighbour by cosine similarity. The
encoder is trained contrastively on MS MARCO (v2.1) query–passage triplets.

The idea: keyword search only matches exact words. Here, text is encoded into a fixed-size vector
(an embedding) such that a query and a passage that *mean* the same thing land close together.
Search then becomes "find the passage whose embedding is nearest to the query's."

## Pipeline

```
MS MARCO (v2.1)
     ↓  build the corpus + (query, positive, hard-negative) triplets
passages.json + train_pairs.json + eval sets
     ↓  train a BPE tokenizer on the corpus + queries
tokenizer.json
     ↓  train the Transformer encoder with a contrastive (triplet InfoNCE) loss
best_model.pt
     ↓  encode all passages → embedding index
     ↓  query time: encode query → cosine similarity → top-K passages
retrieval
```

Two notebooks run this end to end: `data-pipeline.ipynb` (builds the data + tokenizer) and
`training.ipynb` (trains the encoder and produces the model). The data and model artifacts are
regenerated on Kaggle and are not versioned in git.

## Model

A from-scratch Transformer encoder (`model/model.py`): token + sinusoidal positional embeddings,
4 stacked self-attention blocks (`d_model=256`, 4 heads, GELU feed-forward), and masked mean
pooling over the final hidden states to produce one 256-d embedding per text. A simpler
`BagOfEmbeddings` encoder is included as a lightweight baseline.

Training uses a contrastive loss (`model/contrastive_learning.py`): each query is pulled toward
its positive passage and pushed away from every other passage in the batch plus an explicit hard
negative. With batch size `B`, every query is scored against `2B` candidates (the batch's
positives + hard negatives), so larger batches mean more negatives and a stronger signal.

## Training experiments

The encoder was trained from scratch in **two runs**. The aim was not to chase the lowest possible
number but to *understand the training dynamics and justify each choice*: train a baseline, read
its loss curve to diagnose what is limiting it, form a hypothesis, change the smallest set of
variables that tests that hypothesis, and re-measure. Run 1 establishes a baseline and diagnoses
the bottleneck; Run 2 acts on that diagnosis.

**How to read the metric.** Validation loss is the contrastive cross-entropy of `triplet_nce`:
each query is scored against `2 × batch_size` candidates (the batch's positives + one hard
negative per example) and must identify its own positive. Two consequences matter for
interpretation:

- A **random** model scores `ln(2·batch_size)` (e.g. `ln(128) ≈ 4.85` at batch 64). So the loss
  is only meaningful relative to that baseline, and `e^(−loss)` is roughly the probability mass
  the model puts on the correct passage — an interpretable "how often is it right" proxy.
- It is an **optimistic** proxy. It measures picking 1-of-`2B` *within a batch*, whereas real
  retrieval picks 1 out of the entire corpus (hundreds of thousands of passages). So a good val
  loss is necessary but not sufficient; the honest verdict needs Recall@K / MRR over the full
  corpus (the evaluation stage), not this number alone.

### Run 1 — baseline

**What I was doing and why.** A deliberately conservative first pass to get a working, converging
training loop and a reference point. Settings were standard contrastive-learning defaults so that
nothing exotic could be blamed later: a *small* 50k-example slice of MS MARCO for fast iteration,
batch 64, `lr 1e-4` with AdamW + warmup + cosine decay, temperature `0.05` (the usual InfoNCE
value), and `max_len 128` (MS MARCO passages are short, ~56 words, so little is truncated).

| Setting       | Value                                         |
|---------------|-----------------------------------------------|
| Train triples | 24,454 (MS MARCO, 50k examples)               |
| Epochs        | 10                                            |
| Batch size    | 64                                            |
| Learning rate | 1e-4 (AdamW, warmup + cosine decay)           |
| Max length    | 128 tokens                                    |
| Temperature   | 0.05                                          |
| Model         | 4-layer Transformer, d_model=256, vocab=8,000 |

**Validation loss per epoch:**

| Epoch    | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10     |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| val_loss | 2.481 | 2.188 | 2.072 | 2.007 | 1.966 | 1.944 | 1.930 | 1.924 | 1.922 | 1.9217 |

**Analysis.** Training converged cleanly — smooth, monotonic, no val-loss bounce, so no
overfitting; the loop and loss are working correctly. The model learned a *real* signal: final
loss 1.92 against the random baseline of 4.85 means it places ~`e^(-1.92) ≈ 15%` of its
probability mass on the correct passage among 128 candidates, ~18× better than the 0.8% of pure
chance. **But the improvements died early** — the curve is essentially flat from epoch 8 (epoch
9→10 improves by 0.0005) and the cosine schedule has annealed the LR to ~0, so the model is not
under-trained; it has extracted what it can. The diagnosis pointed at two concrete bottlenecks:
(1) only 24,454 training triples — **3%** of MS MARCO's 808k train split, i.e. severely
data-starved for a from-scratch model; and (2) batch 64 gives each query only **127** in-batch
negatives, and contrastive learning gets its signal from the number and difficulty of negatives.

### Run 2 — acting on the diagnosis

**Hypothesis.** If Run 1 is limited by data quantity and by too few negatives, then increasing
both should lower the loss and, more importantly, produce better embeddings. The changes:

- **4× more data** (50k → 200k MS MARCO examples → 97,884 triples) — directly addresses the
  data-starvation bottleneck; this was expected to be the dominant lever.
- **Batch 64 → 128** — doubles in-batch negatives (127 → 255), the main driver of contrastive
  signal.
- **LR 1e-4 → 2e-4** — scaled with batch size (the standard linear-scaling heuristic: a 2× batch
  averages over more examples, so a larger step is warranted).
- **Epochs 10 → 15** — a bigger batch means fewer gradient steps per epoch, so epochs were raised
  to keep the *total update count* comparable and avoid confounding "more data" with "fewer
  updates".

*Honest caveat:* this changes four things at once, so it is not a clean single-variable ablation —
the gain can't be attributed to one factor in isolation. Given Run 1's diagnosis (data was the
clearest deficit) the data increase is the most likely dominant cause, but this is a reasoned
attribution, not a proven one.

| Setting           | Run 1  | Run 2      |
|-------------------|--------|------------|
| Train triples     | 24,454 | **97,884** |
| Batch size        | 64     | **128**    |
| Learning rate     | 1e-4   | **2e-4**   |
| Epochs            | 10     | **15**     |
| Negatives / query | 127    | **255**    |

**Validation loss per epoch:**

| Epoch    | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    | 11    | 12    | 13    | 14     | 15     |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| val_loss | 2.338 | 1.977 | 1.801 | 1.705 | 1.628 | 1.577 | 1.544 | 1.519 | 1.497 | 1.482 | 1.475 | 1.470 | 1.467 | 1.4653 | 1.4652 |

|                              | Run 1  | Run 2      |
|------------------------------|--------|------------|
| Best val_loss                | 1.9217 | **1.4652** |
| Candidates in loss (2×batch) | 128    | 256        |
| Random baseline `ln(2·bs)`   | 4.85   | 5.55       |
| P(correct) = `e^(−loss)`     | ~14.6% | **~23.0%** |
| Final train_loss             | —      | 1.18       |

**Analysis.** The hypothesis held, and the improvement is *larger than the raw numbers suggest*.
Run 2 reached **1.4652 vs Run 1's 1.9217 — but on a harder objective**: doubling the batch raises
the in-batch candidate count from 128 to 256, so the model is discriminating among twice as many
passages and *still* scores lower. Normalising for that, the probability mass on the correct
passage rose from ~15% to ~23% (and relative to each run's own random baseline, from 2.5 nats
below chance to 4.1 nats below). It converged just as cleanly (epoch 14→15 moved 0.0001), and the
train/val gap (1.18 vs 1.47) is modest — the larger dataset suppressed overfitting rather than
causing it. So more data plus more negatives produced measurably and meaningfully better
embeddings.

### Final assessment

- **The intervention worked.** Iterating from a diagnosed bottleneck to a targeted fix improved
  the model on every comparable measure, and by a clear margin (≈24% lower loss under a harder
  objective; ~15% → ~23% top-passage probability). This is the central result.
- **The model is now capacity-bound, not data-bound.** Run 2 plateaued exactly as Run 1 did
  (flat from ~epoch 12). Having removed the data bottleneck, the new ceiling is the architecture
  itself — a 4-layer, `d_model=256` encoder. Within this architecture, more data or epochs would
  yield diminishing returns; the next lever would be **model capacity** (deeper / wider) or
  harder-negative mining. This is left as future work rather than pursued, because the project's
  goal is understanding and justified iteration, not squeezing the last decimal of loss.
- **Why I stopped at Run 2.** Three reasons agree: the run had converged (no value in more
  epochs), the diagnosis showed the remaining gap is architectural (out of scope for a tuning
  change), and the most valuable remaining work is evaluation, not further training.
- **The honest limit of this verdict.** Everything above is measured in *validation loss*, an
  in-batch proxy. It establishes that training is healthy and that Run 2's embeddings are better
  than Run 1's, but it does **not** establish real-world retrieval quality. The definitive
  judgement — does the encoder beat TF-IDF / BM25 on Recall@K and MRR over the full corpus —
  belongs to the evaluation stage, and that comparison is where the model's usefulness is actually
  decided. `best_model.pt` (Run 2) is the model carried forward to that stage.

## Note on the corpus size

Run 2 also grows the passage corpus (~488k → ~1.9M passages). Absolute Recall@K drops for *any*
retriever when the corpus grows, simply because there are more distractors — so retrieval numbers
should always be compared against baselines (TF-IDF, BM25) evaluated on the *same* corpus, not
read as absolute scores.
