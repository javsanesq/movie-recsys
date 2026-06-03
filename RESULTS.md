# Offline Evaluation Results

**Dataset:** MovieLens 25M. **Split:** temporal — train `< 2017-01-01`, validation `2017`, test `≥ 2018-01-01`.
**Protocol:** for each test user with ≥ 3 positive (rating ≥ 4.0) test items, generate top-K recommendations and score against held-out test positives. Already-rated (train) items are excluded from candidates. **Users evaluated: 3,646.**

## Model comparison

| Model | NDCG@5 | NDCG@10 | Recall@10 | Precision@10 | MAP@10 | Coverage@10 | Personalization@10 |
|---|---|---|---|---|---|---|---|
| Popularity        | 0.0804 | 0.0724 | 0.0285 | 0.0614 | 0.0355 | 0.0155 | 0.835 |
| ALS (Stage 1)     | 0.0656 | 0.0662 | 0.0291 | 0.0621 | 0.0291 | 0.2161 | 0.996 |
| **ALS + LambdaRank (both stages)** | **0.1230** | **0.1148** | **0.0495** | **0.0991** | **0.0589** | 0.0695 | 0.962 |

**The two-stage system wins on every accuracy metric** — NDCG@10 is **+58%** over the popularity baseline and **+73%** over ALS-only; Precision@10 and Recall@10 roughly **double**.

### Why is ALS-only *below* popularity on NDCG@10?

This is expected and informative. ALS retrieval is tuned as a **recall-oriented candidate generator**, not a precision-oriented ranker — it casts a wide, diverse net (note its Coverage@10 of 0.22 and near-perfect Personalization, vs popularity's 0.015 coverage). Its raw top-10 ordering is worse than simply showing globally popular movies. The **Stage-2 LambdaRank re-ranker is exactly what fixes this**: it reorders the 200 ALS candidates using user/item/cross features and beats *both* baselines. That is the entire argument for a two-stage architecture.

## Feature importance (LightGBM gain, top 10)

| Feature | Share |
|---|---|
| `n_ratings` (item popularity) | 0.354 |
| `genre_match` (user affinity · item genres) | 0.111 |
| `avg_rating` (Bayesian item mean) | 0.066 |
| `days_since_last` (user recency) | 0.063 |
| `user_item_era_diff` | 0.040 |
| `u_n_ratings` (user activity) | 0.032 |
| `genre_affinity_IMAX` | 0.022 |
| `genre_affinity_Crime` | 0.021 |
| `genre_affinity_Adventure` | 0.012 |
| `genre_affinity_Action` | 0.011 |

Item popularity dominates but does **not** monopolize — the learned `genre_match` cross-feature and per-user recency/affinity signals contribute meaningfully, which is why the re-ranker personalizes rather than collapsing to a popularity list. No single feature approaches 1.0 (a sign there is no target leak).

## Stage-1 retrieval sanity check

**Recall@200 = 0.297** on the validation set (fraction of a user's future positive items captured in the 200 ALS candidates). Two choices drove this up from a 0.187 naive baseline:

- **Raw inner product, not cosine.** L2-normalizing the vectors optimizes cosine similarity, but ALS is trained on the dot product. Searching with the raw inner product matches the training objective (+~1 pt, and it is the correct thing to do).
- **Excluding already-seen items.** Filtering each user's train-rated movies out of the candidate set frees slots for novel items (+8 pts).

> The original plan targeted Recall@200 ≥ 0.50. That number was unvalidated; on a strict temporal split where evaluation items are *future* ratings, ~0.30 is the honest, healthy result. Reported as-is rather than reverse-engineering the target.

## Design decisions & trade-offs

- **Temporal split, not random.** A random split leaks the future: a movie a user rates in 2018 must not shape recommendations evaluated at 2016. All user/item features are computed on the train period only.
- **ALS confidence weighting.** Implicit feedback is binarized and weighted `confidence = 1 + α·rating` (α=10): a 5-star interaction carries more weight than a 1-star one, without treating absence as a hard negative.
- **LambdaRank, not pointwise regression.** LambdaRank optimizes NDCG directly (list-level rank quality), whereas pointwise MSE on ratings ignores how the ranking is ordered.
- **Brute-force search instead of Faiss.** At ~18k items, exact `item_vectors @ user_vec` top-k is ~2 ms — an ANN index adds no value at this scale. It also avoids a hard blocker: `faiss-cpu` and `lightgbm` each statically link their own OpenMP runtime, which **deadlocks** when both are imported in one process (the eval and serving paths do exactly that on macOS/Apple-Silicon).
- **8 GB memory budget.** The ranker is trained on a 20k-user sample (≈4M candidate rows); genome tags are streamed and reduced to the top-20 before pivoting; evaluation runs on a capped user sample. Every compromise is deliberate and documented.
- **Cold-start.** Users/items below the support thresholds (≥5 / ≥10 train ratings) are excluded from evaluation. A production system would fall back to popularity + content-based recommendations for them.
