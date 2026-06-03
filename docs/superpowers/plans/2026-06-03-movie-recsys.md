# movie-recsys Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage movie recommender (ALS+Faiss retrieval → LightGBM LambdaRank re-rank) on MovieLens 25M, served by FastAPI behind a vanilla-JS dashboard, deployed with Docker Compose, with a real offline evaluation and an incremental `.docx` study guide.

**Architecture:** Standalone pipeline scripts turn raw ml-25m into Parquet features (no temporal leakage). `train/` produces `als.pkl` + `faiss.index` (Stage 1) and `ranker.txt` (Stage 2). `eval/` writes `artifacts/metrics.json` comparing popularity / ALS-only / ALS+LambdaRank. `api/` loads all artifacts once at startup and serves recommendations in <200ms. `ui/` is a 3-panel vanilla-JS dashboard behind nginx. Docker Compose runs `api` (8000) + `ui` (80).

**Tech Stack:** Python 3.11 (via `uv`), Polars, pandas, scipy, implicit (ALS), faiss-cpu, LightGBM, FastAPI, uvicorn, joblib, python-docx, vanilla JS, nginx, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-06-03-movie-recsys-design.md`

**Hard constraints (8 GB RAM, M2):** Stream genome with Polars and filter to top-20 tags *before* pivoting. Sample ~40k train users for the ranker matrix. Cap eval at ~5k test users. Cast all Faiss inputs to `np.float32`. Never random-split — temporal only.

**Commit policy:** Author = `Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>`, **no Co-Authored-By trailer**. One commit per completed phase.

---

## File Structure

```
movie-recsys/
├── pyproject.toml              # uv project + deps (py3.11)
├── requirements.txt            # exported for Dockerfile pip install
├── .gitignore                  # raw data, artifacts (except metrics.json), venv
├── Makefile                    # setup/pipeline/train/eval/api/up targets
├── README.md, RESULTS.md, SECURITY.md, LICENSE
├── config.py                   # shared constants (paths, split boundaries, thresholds)
├── data/download.py            # fetch + unzip ml-25m → data/raw/ml-25m/
├── pipeline/
│   ├── split.py                # temporal split → data/splits/{train,val,test}.parquet
│   ├── build_genome_pivot.py   # streaming top-20 genome tags → genome_top20.parquet
│   ├── item_features.py        # per-item features (+ genome merge) → items.parquet
│   └── user_features.py        # per-user features → users.parquet
├── train/
│   ├── retrieval.py            # ALS + Faiss index + id maps
│   ├── ranker.py               # LambdaRank training data + model
│   └── baselines.py            # popularity baseline candidate list
├── eval/
│   ├── metrics.py              # ndcg@k, recall@k, precision@k, map@k, coverage, personalization
│   └── evaluate.py             # 3-way offline eval → artifacts/metrics.json
├── api/{main.py, recommender.py, schemas.py}
├── ui/{index.html, app.js, style.css, nginx.conf}
├── tests/{test_metrics.py, test_features.py, test_split.py, test_api.py}
├── artifacts/{models/, features/, metrics.json}
├── output/doc/movie-recsys-technical-deep-dive.docx
├── Dockerfile.api, docker-compose.yml
```

**Genres (canonical 20, used everywhere):** `Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, (no genres listed)`. Defined once in `config.py` as `GENRES`.

---

## Phase 1: Scaffold + Data Pipeline

**Verification gate:** splits + 3 parquet feature files exist with printed row counts; feature tests pass; no temporal leakage.

### Task 1.1: Project scaffold, uv env, gitignore

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `config.py`, `LICENSE`, `SECURITY.md`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "movie-recsys"
version = "0.1.0"
description = "Two-stage movie recommender: ALS+Faiss retrieval, LightGBM LambdaRank re-rank"
requires-python = ">=3.11,<3.12"
dependencies = [
    "polars>=1.0",
    "pandas>=2.2",
    "pyarrow>=16",
    "numpy>=1.26,<2.0",
    "scipy>=1.13",
    "implicit>=0.7.2",
    "faiss-cpu>=1.8",
    "lightgbm>=4.3",
    "fastapi>=0.111",
    "uvicorn>=0.30",
    "joblib>=1.4",
    "python-docx>=1.1",
    "requests>=2.32",
]

[dependency-groups]
dev = ["pytest>=8", "httpx>=0.27"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Create venv and install with uv**

Run: `cd ~/Desktop/Proyectos/movie-recsys && uv python install 3.11 && uv sync`
Expected: `.venv/` created with Python 3.11; faiss-cpu, lightgbm, implicit resolve to arm64 wheels. No build errors.

- [ ] **Step 3: Smoke-test the heavy imports**

Run: `uv run python -c "import faiss, lightgbm, implicit, scipy.sparse, polars; print('imports ok', faiss.__version__)"`
Expected: `imports ok 1.x.x` with no segfault.

- [ ] **Step 4: Create `config.py`**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "ml-25m"
SPLITS = ROOT / "data" / "splits"
FEATURES = ROOT / "artifacts" / "features"
MODELS = ROOT / "artifacts" / "models"
METRICS_JSON = ROOT / "artifacts" / "metrics.json"

# Temporal split boundaries (Unix epoch, UTC)
TRAIN_END = 1483228800  # 2017-01-01
VAL_END = 1514764800    # 2018-01-01

# Filtering thresholds
MIN_USER_TRAIN_RATINGS = 5
MIN_ITEM_TRAIN_RATINGS = 10

# Modeling
POSITIVE_THRESHOLD = 4.0       # rating >= this = relevant (eval)
N_CANDIDATES = 200             # Faiss top-N retrieval
ALS_FACTORS = 128
ALS_ITERATIONS = 20
ALS_REGULARIZATION = 0.01
ALS_ALPHA = 10.0               # confidence = 1 + alpha * rating
RANKER_USER_SAMPLE = 40000     # users sampled for ranker training (8 GB limit)
EVAL_USER_SAMPLE = 5000        # test users evaluated
GENOME_TOP_TAGS = 20
BAYES_MIN_COUNT = 25           # shrinkage prior count for item avg rating

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
    "Western", "(no genres listed)",
]

for _d in (SPLITS, FEATURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.DS_Store
data/raw/
data/splits/
artifacts/models/
artifacts/features/
# keep metrics.json (committed)
!artifacts/metrics.json
ml-25m.zip
```

- [ ] **Step 6: Create `LICENSE` (MIT) and `SECURITY.md`**

Copy MIT license text with `Copyright (c) 2026 Javier Sánchez Esquivel`. `SECURITY.md`: brief — this is a local demo with no auth; report issues via GitHub issues; no secrets stored; artifacts are derived from public MovieLens data.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore config.py LICENSE SECURITY.md
git commit --author="Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>" -m "Scaffold: uv project, config, license"
```

### Task 1.2: Dataset download

**Files:**
- Create: `data/download.py`

- [ ] **Step 1: Write `data/download.py`**

```python
"""Download and extract MovieLens 25M into data/raw/ml-25m/."""
import sys, zipfile, urllib.request
from pathlib import Path
from config import DATA_RAW

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_PATH = DATA_RAW.parent / "ml-25m.zip"

def main():
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    expected = ["ratings.csv", "movies.csv", "genome-scores.csv", "genome-tags.csv"]
    if all((DATA_RAW / f).exists() for f in expected):
        print("ml-25m already present.")
        return
    if not ZIP_PATH.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_RAW.parent)
    for f in expected:
        assert (DATA_RAW / f).exists(), f"missing {f} after extract"
    print("Done:", DATA_RAW)

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run download**

Run: `uv run python data/download.py`
Expected: ~250 MB download, extraction, prints `Done: .../data/raw/ml-25m`. `ls data/raw/ml-25m/` shows the 4 CSVs.

### Task 1.3: Temporal split (TDD)

**Files:**
- Create: `pipeline/split.py`, `tests/test_split.py`

- [ ] **Step 1: Write failing test for `assign_split`**

```python
# tests/test_split.py
import polars as pl
from pipeline.split import assign_split
from config import TRAIN_END, VAL_END

def test_assign_split_boundaries():
    df = pl.DataFrame({"timestamp": [TRAIN_END - 1, TRAIN_END, VAL_END - 1, VAL_END]})
    out = assign_split(df)["split"].to_list()
    assert out == ["train", "val", "val", "test"]
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_split.py -v`
Expected: FAIL — `ImportError: cannot import name 'assign_split'`.

- [ ] **Step 3: Implement `pipeline/split.py`**

```python
"""Temporal train/val/test split + filtering. Streams with Polars."""
import polars as pl
from config import (DATA_RAW, SPLITS, TRAIN_END, VAL_END,
                    MIN_USER_TRAIN_RATINGS, MIN_ITEM_TRAIN_RATINGS)

def assign_split(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("timestamp") < TRAIN_END).then(pl.lit("train"))
          .when(pl.col("timestamp") < VAL_END).then(pl.lit("val"))
          .otherwise(pl.lit("test")).alias("split")
    )

def main():
    ratings = pl.read_csv(DATA_RAW / "ratings.csv")
    ratings = assign_split(ratings)
    train = ratings.filter(pl.col("split") == "train")

    # Keep users/items with enough TRAIN support
    good_users = (train.group_by("userId").len()
                  .filter(pl.col("len") >= MIN_USER_TRAIN_RATINGS)["userId"])
    good_items = (train.group_by("movieId").len()
                  .filter(pl.col("len") >= MIN_ITEM_TRAIN_RATINGS)["movieId"])
    ratings = ratings.filter(
        pl.col("userId").is_in(good_users) & pl.col("movieId").is_in(good_items))

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        part = ratings.filter(pl.col("split") == name)
        part.write_parquet(SPLITS / f"{name}.parquet")
        print(f"{name}: {part.height:,} rows, "
              f"{part['userId'].n_unique():,} users, {part['movieId'].n_unique():,} items")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_split.py -v`
Expected: PASS.

- [ ] **Step 5: Run the split**

Run: `uv run python pipeline/split.py`
Expected: prints row/user/item counts per split; `data/splits/{train,val,test}.parquet` exist. Train should be the large majority.

### Task 1.4: Genome pivot — streaming top-20 (memory-critical)

**Files:**
- Create: `pipeline/build_genome_pivot.py`

- [ ] **Step 1: Write `pipeline/build_genome_pivot.py`**

```python
"""Top-20 genome tags pivoted per movie. Filters tags BEFORE pivoting (8 GB safe)."""
import polars as pl
from config import DATA_RAW, FEATURES, GENOME_TOP_TAGS

def main():
    scores = DATA_RAW / "genome-scores.csv"
    # 1) mean relevance per tag via streaming scan (no full materialization)
    top = (pl.scan_csv(scores)
           .group_by("tagId").agg(pl.col("relevance").mean().alias("m"))
           .sort("m", descending=True).head(GENOME_TOP_TAGS)
           .collect(streaming=True))
    top_ids = top["tagId"].to_list()
    tags = pl.read_csv(DATA_RAW / "genome-tags.csv").filter(pl.col("tagId").is_in(top_ids))
    id_to_name = {r["tagId"]: r["tag"] for r in tags.iter_rows(named=True)}

    # 2) filter to top tags FIRST, then pivot (small)
    sub = (pl.scan_csv(scores).filter(pl.col("tagId").is_in(top_ids))
           .collect(streaming=True))
    pivot = sub.pivot(values="relevance", index="movieId", on="tagId")
    pivot = pivot.rename({str(t): f"genome_{id_to_name[t]}" for t in top_ids if str(t) in pivot.columns})
    pivot.write_parquet(FEATURES / "genome_top20.parquet")
    print(f"genome_top20: {pivot.shape}, tags={list(id_to_name.values())}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and watch memory**

Run: `/usr/bin/time -l uv run python pipeline/build_genome_pivot.py 2>&1 | tail -5`
Expected: prints shape ~`(13816, 21)` and the 20 tag names; peak RSS well under 4 GB (the filter-first approach keeps it small). If RSS spikes near 8 GB, stop — the streaming filter isn't engaging.

### Task 1.5: Item features (TDD for the testable bits)

**Files:**
- Create: `pipeline/item_features.py`, `tests/test_features.py`

- [ ] **Step 1: Write failing tests for helpers**

```python
# tests/test_features.py
import polars as pl
from pipeline.item_features import extract_year, bayesian_avg

def test_extract_year():
    s = pl.Series(["Toy Story (1995)", "No Year Movie", "Blade Runner (1982)"])
    assert extract_year(s).to_list() == [1995, None, 1982]

def test_bayesian_avg_shrinks_low_count():
    # count=1, local mean 5.0, global 3.0, prior 25 -> close to global
    val = bayesian_avg(local_mean=5.0, count=1, global_mean=3.0, prior=25)
    assert 3.0 < val < 3.2
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `pipeline/item_features.py`**

```python
"""Per-item features computed on TRAIN split only, merged with genome top-20."""
import polars as pl
from config import (DATA_RAW, SPLITS, FEATURES, GENRES, BAYES_MIN_COUNT)

def extract_year(titles: pl.Series) -> pl.Series:
    return titles.str.extract(r"\((\d{4})\)", 1).cast(pl.Int32, strict=False)

def bayesian_avg(local_mean: float, count: int, global_mean: float, prior: int) -> float:
    return (count * local_mean + prior * global_mean) / (count + prior)

def main():
    train = pl.read_parquet(SPLITS / "train.parquet")
    movies = pl.read_csv(DATA_RAW / "movies.csv")
    global_mean = train["rating"].mean()

    agg = train.group_by("movieId").agg(
        pl.len().alias("n_ratings"),
        pl.col("rating").mean().alias("local_mean"),
        pl.col("rating").std().fill_null(0.0).alias("std_rating"),
    )
    agg = agg.with_columns(
        ((pl.col("n_ratings") * pl.col("local_mean") + BAYES_MIN_COUNT * global_mean)
         / (pl.col("n_ratings") + BAYES_MIN_COUNT)).alias("avg_rating")
    ).drop("local_mean")

    items = movies.join(agg, on="movieId", how="inner")
    items = items.with_columns(extract_year(pl.col("title")).alias("release_year"))
    for g in GENRES:
        items = items.with_columns(
            pl.col("genres").str.contains(g, literal=True).cast(pl.Int8).alias(f"genre_{g}"))

    genome = pl.read_parquet(FEATURES / "genome_top20.parquet")
    items = items.join(genome, on="movieId", how="left").fill_null(0.0)
    items.write_parquet(FEATURES / "items.parquet")
    print(f"items: {items.shape}, global_mean={global_mean:.3f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: PASS (both).

- [ ] **Step 5: Build item features**

Run: `uv run python pipeline/item_features.py`
Expected: prints shape (rows ≈ count of items meeting MIN_ITEM_TRAIN_RATINGS, cols = base + 20 genre + 20 genome). `items.parquet` exists.

### Task 1.6: User features

**Files:**
- Create: `pipeline/user_features.py`

- [ ] **Step 1: Implement `pipeline/user_features.py`**

```python
"""Per-user features computed on TRAIN split only."""
import polars as pl
from config import DATA_RAW, SPLITS, FEATURES, GENRES, TRAIN_END

def main():
    train = pl.read_parquet(SPLITS / "train.parquet")
    movies = pl.read_csv(DATA_RAW / "movies.csv").select("movieId", "genres")
    t = train.join(movies, on="movieId", how="left")

    base = train.group_by("userId").agg(
        pl.len().alias("n_ratings"),
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").std().fill_null(0.0).alias("std_rating"),
        pl.col("timestamp").max().alias("last_ts"),
        (pl.col("rating") >= 4.0).mean().alias("pct_high_ratings"),
    ).with_columns(
        ((TRAIN_END - pl.col("last_ts")) / 86400.0).alias("days_since_last")
    ).drop("last_ts")

    # genre affinity: fraction of a user's rated movies in each genre
    for g in GENRES:
        ga = (t.with_columns(pl.col("genres").str.contains(g, literal=True).cast(pl.Int8).alias("hit"))
              .group_by("userId").agg(pl.col("hit").mean().alias(f"genre_affinity_{g}")))
        base = base.join(ga, on="userId", how="left")

    base = base.fill_null(0.0)
    base.write_parquet(FEATURES / "users.parquet")
    print(f"users: {base.shape}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build user features**

Run: `uv run python pipeline/user_features.py`
Expected: prints shape (cols = 6 base + 20 genre_affinity). `users.parquet` exists.

- [ ] **Step 3: Leakage assertion**

Run: `uv run python -c "import polars as pl; from config import SPLITS, VAL_END; t=pl.read_parquet(SPLITS/'train.parquet'); assert t['timestamp'].max() < VAL_END; print('no leakage: train max ts < VAL_END')"`
Expected: prints the no-leakage confirmation.

- [ ] **Step 4: Commit Phase 1**

```bash
git add config.py data/ pipeline/ tests/test_split.py tests/test_features.py
git commit --author="Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>" -m "Phase 1: data download + temporal split + feature pipeline"
```

---

## Phase 2: ALS Retrieval + Faiss

**Verification gate:** `als.pkl`, `faiss.index`, `idx_to_movieid.npy`, `idx_to_userid.npy` exist; **Recall@200 ≥ 0.50** on val.

### Task 2.1: ALS training + Faiss index + id maps

**Files:**
- Create: `train/retrieval.py`

- [ ] **Step 1: Implement `train/retrieval.py`**

```python
"""Stage 1: ALS confidence model -> item vectors -> Faiss IndexFlatIP."""
import numpy as np, joblib, faiss, polars as pl
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from config import (SPLITS, MODELS, ALS_FACTORS, ALS_ITERATIONS,
                    ALS_REGULARIZATION, ALS_ALPHA, N_CANDIDATES, POSITIVE_THRESHOLD)

def build_matrix():
    train = pl.read_parquet(SPLITS / "train.parquet")
    users = train["userId"].unique().sort().to_list()
    items = train["movieId"].unique().sort().to_list()
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {m: i for i, m in enumerate(items)}
    rows = train["userId"].replace_strict(uidx).to_numpy()
    cols = train["movieId"].replace_strict(iidx).to_numpy()
    conf = (1.0 + ALS_ALPHA * train["rating"].to_numpy()).astype(np.float32)
    mat = csr_matrix((conf, (rows, cols)), shape=(len(users), len(items)))
    return mat, np.array(users), np.array(items), uidx, iidx

def main():
    user_items, idx_to_userid, idx_to_movieid, uidx, iidx = build_matrix()
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS, iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION, use_gpu=False)
    model.fit(user_items)

    item_vecs = np.ascontiguousarray(model.item_factors.astype(np.float32))
    faiss.normalize_L2(item_vecs)
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(item_vecs)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "als.pkl")
    faiss.write_index(index, str(MODELS / "faiss.index"))
    np.save(MODELS / "idx_to_movieid.npy", idx_to_movieid)
    np.save(MODELS / "idx_to_userid.npy", idx_to_userid)

    recall = eval_recall(model, index, user_items, idx_to_movieid, idx_to_userid, uidx, iidx)
    print(f"Recall@{N_CANDIDATES} = {recall:.4f}")

def eval_recall(model, index, user_items, idx_to_movieid, idx_to_userid, uidx, iidx, sample=5000):
    val = pl.read_parquet(SPLITS / "val.parquet").filter(pl.col("rating") >= POSITIVE_THRESHOLD)
    val = val.filter(pl.col("userId").is_in(list(uidx.keys())))
    movieid_set = set(idx_to_movieid.tolist())
    hits = total = 0
    user_pos = val.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    rng = np.random.default_rng(0)
    pairs = list(zip(user_pos["userId"], user_pos["movieId"]))
    rng.shuffle(pairs)
    for uid, pos in pairs[:sample]:
        pos = [m for m in pos if m in movieid_set]
        if not pos:
            continue
        uvec = model.user_factors[uidx[uid]].astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(uvec)
        _, I = index.search(uvec, N_CANDIDATES)
        retrieved = set(idx_to_movieid[I[0]].tolist())
        hits += sum(1 for m in pos if m in retrieved)
        total += len(pos)
    return hits / max(total, 1)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train ALS + build index**

Run: `uv run python train/retrieval.py`
Expected: implicit shows a progress bar; prints `Recall@200 = 0.XX`. Artifacts written. **Gate: Recall ≥ 0.50.** If below, raise `ALS_FACTORS` to 192 or `ALS_ITERATIONS` to 30 and rerun; note the change.

- [ ] **Step 3: Verify artifacts**

Run: `ls -la artifacts/models/`
Expected: `als.pkl`, `faiss.index`, `idx_to_movieid.npy`, `idx_to_userid.npy`.

- [ ] **Step 4: Commit Phase 2**

```bash
git add train/retrieval.py
git commit --author="Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>" -m "Phase 2: ALS retrieval + Faiss index, Recall@200 gate"
```

---

## Phase 3: LightGBM LambdaRank + Baselines + Offline Eval

**Verification gate:** `ranker.txt` exists; `artifacts/metrics.json` shows `als_lgbm` NDCG@10 > `als_only` > `popularity`.

### Task 3.1: Ranking metrics (pure TDD)

**Files:**
- Create: `eval/metrics.py`, `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import numpy as np
from eval.metrics import ndcg_at_k, recall_at_k, precision_at_k, average_precision_at_k

def test_ndcg_perfect_is_one():
    ranked = [10, 20, 30]; relevant = {10, 20, 30}
    assert abs(ndcg_at_k(ranked, relevant, 3) - 1.0) < 1e-9

def test_ndcg_reversed_less_than_one():
    ranked = [99, 98, 10]; relevant = {10}
    assert 0.0 < ndcg_at_k(ranked, relevant, 3) < 1.0

def test_recall_half():
    assert recall_at_k([1, 2, 9, 9], {1, 2, 3, 4}, 4) == 0.5

def test_precision_quarter():
    assert precision_at_k([1, 9, 9, 9], {1, 2}, 4) == 0.25

def test_average_precision():
    # hits at ranks 1 and 3 -> (1/1 + 2/3)/2
    ap = average_precision_at_k([1, 9, 2, 9], {1, 2}, 4)
    assert abs(ap - (1.0 + 2/3) / 2) < 1e-9
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `eval/metrics.py`**

```python
"""Ranking metrics. `ranked` = list of item ids (best first); `relevant` = set of ids."""
import numpy as np

def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))

def ndcg_at_k(ranked, relevant, k):
    gains = [1.0 if item in relevant else 0.0 for item in ranked[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0

def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)

def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k

def average_precision_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    score = hits = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (all 5).

### Task 3.2: Shared feature-matrix builder

**Files:**
- Create: `eval/featbuild.py` (shared by ranker training, eval, and API)

- [ ] **Step 1: Implement `eval/featbuild.py`**

```python
"""Build (user x candidate) feature matrix. Shared across train/eval/serve so
features are identical everywhere (no train/serve skew)."""
import numpy as np, polars as pl
from config import GENRES

GENRE_COLS = [f"genre_{g}" for g in GENRES]
AFF_COLS = [f"genre_affinity_{g}" for g in GENRES]

def feature_columns(items_df: pl.DataFrame) -> list[str]:
    item_cols = (["n_ratings", "avg_rating", "std_rating", "release_year"]
                 + GENRE_COLS + [c for c in items_df.columns if c.startswith("genome_")])
    user_cols = (["u_n_ratings", "u_avg_rating", "u_std_rating",
                  "days_since_last", "pct_high_ratings"] + AFF_COLS)
    cross_cols = ["genre_match", "rating_delta", "popularity_rank", "user_item_era_diff"]
    return item_cols + user_cols + cross_cols

def build_matrix(user_row: dict, cand_movieids: np.ndarray,
                 items_df: pl.DataFrame, max_item_ratings: int) -> tuple[np.ndarray, list[str]]:
    cands = items_df.filter(pl.col("movieId").is_in(cand_movieids.tolist()))
    # preserve candidate order
    order = pl.DataFrame({"movieId": cand_movieids, "_o": np.arange(len(cand_movieids))})
    cands = cands.join(order, on="movieId", how="inner").sort("_o").drop("_o")

    aff = np.array([user_row.get(c, 0.0) for c in AFF_COLS], dtype=np.float32)
    genre_mat = cands.select(GENRE_COLS).to_numpy().astype(np.float32)
    genre_match = genre_mat @ aff

    n = cands.height
    feat = {
        "n_ratings": cands["n_ratings"].to_numpy(),
        "avg_rating": cands["avg_rating"].to_numpy(),
        "std_rating": cands["std_rating"].to_numpy(),
        "release_year": cands["release_year"].fill_null(0).to_numpy(),
    }
    for c in GENRE_COLS:
        feat[c] = cands[c].to_numpy()
    for c in cands.columns:
        if c.startswith("genome_"):
            feat[c] = cands[c].to_numpy()
    feat["u_n_ratings"] = np.full(n, user_row["n_ratings"])
    feat["u_avg_rating"] = np.full(n, user_row["avg_rating"])
    feat["u_std_rating"] = np.full(n, user_row["std_rating"])
    feat["days_since_last"] = np.full(n, user_row["days_since_last"])
    feat["pct_high_ratings"] = np.full(n, user_row["pct_high_ratings"])
    for c in AFF_COLS:
        feat[c] = np.full(n, user_row.get(c, 0.0))
    feat["genre_match"] = genre_match
    feat["rating_delta"] = cands["avg_rating"].to_numpy() - user_row["avg_rating"]
    feat["popularity_rank"] = cands["n_ratings"].to_numpy() / max(max_item_ratings, 1)
    user_year = 2017 - (user_row["days_since_last"] / 365.0)
    feat["user_item_era_diff"] = np.abs(cands["release_year"].fill_null(0).to_numpy() - user_year)

    cols = feature_columns(items_df)
    X = np.column_stack([np.asarray(feat[c], dtype=np.float32) for c in cols])
    return X, cands["movieId"].to_list()
```

> Note: `users.parquet` columns `n_ratings/avg_rating/std_rating` are renamed to
> `u_*` at load time in ranker/eval/api to avoid colliding with item columns of the
> same name. The `user_row` dict passed here uses the original user keys
> (`n_ratings`, `avg_rating`, `std_rating`, `days_since_last`, `pct_high_ratings`,
> `genre_affinity_*`).

### Task 3.3: Ranker training data + LambdaRank

**Files:**
- Create: `train/ranker.py`

- [ ] **Step 1: Implement `train/ranker.py`**

```python
"""Stage 2: build training data from ALS candidates, train LightGBM LambdaRank.
Samples RANKER_USER_SAMPLE users to fit 8 GB."""
import numpy as np, joblib, faiss, lightgbm as lgb, polars as pl
from config import (SPLITS, MODELS, FEATURES, N_CANDIDATES, RANKER_USER_SAMPLE,
                    POSITIVE_THRESHOLD)
from eval.featbuild import build_matrix, feature_columns

def load_serving_state():
    model = joblib.load(MODELS / "als.pkl")
    index = faiss.read_index(str(MODELS / "faiss.index"))
    idx_to_movieid = np.load(MODELS / "idx_to_movieid.npy")
    idx_to_userid = np.load(MODELS / "idx_to_userid.npy")
    uidx = {u: i for i, u in enumerate(idx_to_userid.tolist())}
    items = pl.read_parquet(FEATURES / "items.parquet")
    users = pl.read_parquet(FEATURES / "users.parquet").rename(
        {"n_ratings": "n_ratings", "avg_rating": "avg_rating", "std_rating": "std_rating"})
    return model, index, idx_to_movieid, uidx, items, users

def candidates_for(model, index, uidx, idx_to_movieid, uid):
    uvec = model.user_factors[uidx[uid]].astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(uvec)
    _, I = index.search(uvec, N_CANDIDATES)
    return idx_to_movieid[I[0]]

def build_dataset(split_name, model, index, uidx, idx_to_movieid, items, users, user_ids, max_item_ratings):
    label_lut = pl.read_parquet(SPLITS / f"{split_name}.parquet")
    user_dict = {r["userId"]: r for r in users.iter_rows(named=True)}
    Xs, ys, groups = [], [], []
    pos = (label_lut.group_by("userId")
           .agg([pl.col("movieId"), pl.col("rating")]).to_dict(as_series=False))
    rating_map = {u: dict(zip(m, r)) for u, m, r in
                  zip(pos["userId"], pos["movieId"], pos["rating"])}
    for uid in user_ids:
        if uid not in user_dict:
            continue
        cands = candidates_for(model, index, uidx, idx_to_movieid, uid)
        X, mids = build_matrix(user_dict[uid], cands, items, max_item_ratings)
        rmap = rating_map.get(uid, {})
        y = np.array([min(4, int(np.floor(rmap.get(m, 0.0)))) for m in mids], dtype=np.int32)
        Xs.append(X); ys.append(y); groups.append(len(mids))
    return np.vstack(Xs), np.concatenate(ys), np.array(groups)

def main():
    model, index, idx_to_movieid, uidx, items, users = load_serving_state()
    max_item_ratings = int(items["n_ratings"].max())
    users = users.rename({"n_ratings": "n_ratings"})  # keep originals for user_row

    rng = np.random.default_rng(0)
    train_uids = pl.read_parquet(SPLITS / "train.parquet")["userId"].unique().to_numpy()
    train_uids = train_uids[np.isin(train_uids, list(uidx.keys()))]
    rng.shuffle(train_uids)
    train_uids = train_uids[:RANKER_USER_SAMPLE]
    val_uids = pl.read_parquet(SPLITS / "val.parquet")["userId"].unique().to_numpy()
    val_uids = val_uids[np.isin(val_uids, list(uidx.keys()))][:5000]

    Xtr, ytr, gtr = build_dataset("train", model, index, uidx, idx_to_movieid, items, users, train_uids, max_item_ratings)
    Xva, yva, gva = build_dataset("val", model, index, uidx, idx_to_movieid, items, users, val_uids, max_item_ratings)
    print(f"train X={Xtr.shape}, val X={Xva.shape}")

    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr)
    dval = lgb.Dataset(Xva, label=yva, group=gva, reference=dtrain)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [5, 10, 20],
        "label_gain": [0, 1, 3, 7, 15], "num_leaves": 63, "learning_rate": 0.05,
        "min_child_samples": 20, "verbose": -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=300, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])
    booster.save_model(str(MODELS / "ranker.txt"))

    # persist feature importance + column order for serving/eval
    cols = feature_columns(items)
    imp = dict(zip(cols, booster.feature_importance(importance_type="gain").tolist()))
    import json
    (MODELS / "feature_importance.json").write_text(json.dumps(imp))
    (MODELS / "feature_columns.json").write_text(json.dumps(cols))
    print("best ndcg@10:", booster.best_score)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train the ranker (memory watch)**

Run: `/usr/bin/time -l uv run python train/ranker.py 2>&1 | tail -15`
Expected: prints train/val X shapes (train ≈ 8M rows × ~70 cols), early-stopping log every 50 rounds, `best ndcg@10`. Peak RSS should stay under ~5 GB. If it approaches 8 GB, lower `RANKER_USER_SAMPLE` to 25000 and note it in RESULTS.md.

- [ ] **Step 3: Verify model + metadata**

Run: `ls artifacts/models/ && uv run python -c "import json; print(len(json.load(open('artifacts/models/feature_columns.json'))), 'features')"`
Expected: `ranker.txt`, `feature_importance.json`, `feature_columns.json` exist; feature count printed.

### Task 3.4: Popularity baseline

**Files:**
- Create: `train/baselines.py`

- [ ] **Step 1: Implement `train/baselines.py`**

```python
"""Popularity baseline: most-rated TRAIN items, descending."""
import numpy as np, polars as pl
from config import SPLITS, MODELS

def popularity_ranking() -> np.ndarray:
    train = pl.read_parquet(SPLITS / "train.parquet")
    pop = (train.group_by("movieId").len().sort("len", descending=True))
    return pop["movieId"].to_numpy()

def main():
    ranking = popularity_ranking()
    np.save(MODELS / "popularity.npy", ranking)
    print(f"popularity ranking: {len(ranking)} items, top5={ranking[:5].tolist()}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

Run: `uv run python train/baselines.py`
Expected: prints count + top-5 movieIds; `popularity.npy` exists.

### Task 3.5: Full offline evaluation → metrics.json

**Files:**
- Create: `eval/evaluate.py`

- [ ] **Step 1: Implement `eval/evaluate.py`**

```python
"""3-way offline eval (popularity / als_only / als_lgbm) -> artifacts/metrics.json."""
import json, numpy as np, joblib, faiss, lightgbm as lgb, polars as pl
from config import (SPLITS, MODELS, FEATURES, METRICS_JSON, N_CANDIDATES,
                    EVAL_USER_SAMPLE, POSITIVE_THRESHOLD)
from eval.featbuild import build_matrix
from eval.metrics import ndcg_at_k, recall_at_k, precision_at_k, average_precision_at_k

def main():
    als = joblib.load(MODELS / "als.pkl")
    index = faiss.read_index(str(MODELS / "faiss.index"))
    ranker = lgb.Booster(model_file=str(MODELS / "ranker.txt"))
    idx_to_movieid = np.load(MODELS / "idx_to_movieid.npy")
    idx_to_userid = np.load(MODELS / "idx_to_userid.npy")
    uidx = {u: i for i, u in enumerate(idx_to_userid.tolist())}
    pop = np.load(MODELS / "popularity.npy")
    items = pl.read_parquet(FEATURES / "items.parquet")
    users = pl.read_parquet(FEATURES / "users.parquet")
    user_dict = {r["userId"]: r for r in users.iter_rows(named=True)}
    max_item_ratings = int(items["n_ratings"].max())

    test = pl.read_parquet(SPLITS / "test.parquet").filter(pl.col("rating") >= POSITIVE_THRESHOLD)
    posdict = test.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    cand_users = [u for u, m in zip(posdict["userId"], posdict["movieId"])
                  if len(m) >= 3 and u in uidx and u in user_dict]
    rng = np.random.default_rng(0); rng.shuffle(cand_users)
    cand_users = cand_users[:EVAL_USER_SAMPLE]
    relevant = {u: set(m) for u, m in zip(posdict["userId"], posdict["movieId"])}

    acc = {m: {"ndcg@5": [], "ndcg@10": [], "recall@10": [], "precision@10": [], "map@10": []}
           for m in ("popularity", "als_only", "als_lgbm")}
    rec_lists = {m: [] for m in acc}

    for uid in cand_users:
        rel = relevant[uid]
        uvec = als.user_factors[uidx[uid]].astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(uvec)
        _, I = index.search(uvec, N_CANDIDATES)
        cands = idx_to_movieid[I[0]]

        ranked = {
            "popularity": pop[:50].tolist(),
            "als_only": cands.tolist(),
        }
        X, mids = build_matrix(user_dict[uid], cands, items, max_item_ratings)
        scores = ranker.predict(X)
        order = np.argsort(scores)[::-1]
        ranked["als_lgbm"] = [mids[i] for i in order]

        for m, r in ranked.items():
            acc[m]["ndcg@5"].append(ndcg_at_k(r, rel, 5))
            acc[m]["ndcg@10"].append(ndcg_at_k(r, rel, 10))
            acc[m]["recall@10"].append(recall_at_k(r, rel, 10))
            acc[m]["precision@10"].append(precision_at_k(r, rel, 10))
            acc[m]["map@10"].append(average_precision_at_k(r, rel, 10))
            rec_lists[m].append(set(r[:10]))

    def coverage(lists):
        seen = set().union(*lists) if lists else set()
        return len(seen) / items.height
    def personalization(lists):
        if len(lists) < 2:
            return 0.0
        s = rng.choice(len(lists), size=min(500, len(lists)), replace=False)
        sims, sub = [], [lists[i] for i in s]
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                u = len(sub[i] | sub[j])
                sims.append(len(sub[i] & sub[j]) / u if u else 0.0)
        return 1.0 - float(np.mean(sims)) if sims else 0.0

    models = {}
    for m in acc:
        models[m] = {k: round(float(np.mean(v)), 4) for k, v in acc[m].items()}
        models[m]["coverage@10"] = round(coverage(rec_lists[m]), 4)
        models[m]["personalization@10"] = round(personalization(rec_lists[m]), 4)

    fi = json.load(open(MODELS / "feature_importance.json"))
    total = sum(fi.values()) or 1.0
    fi_norm = dict(sorted(((k, round(v / total, 4)) for k, v in fi.items()),
                          key=lambda x: -x[1])[:10])
    out = {"n_users_evaluated": len(cand_users), "models": models, "feature_importance": fi_norm}
    METRICS_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(models, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run evaluation**

Run: `uv run python eval/evaluate.py`
Expected: prints a per-model metric block and writes `artifacts/metrics.json`. **Gate: `als_lgbm` ndcg@10 > `als_only` ndcg@10 > `popularity` ndcg@10.** If the ranker doesn't beat ALS-only, inspect feature_importance (a single dominant feature signals a leak) before tuning.

- [ ] **Step 3: Assert ordering programmatically**

Run: `uv run python -c "import json; m=json.load(open('artifacts/metrics.json'))['models']; a=m['als_lgbm']['ndcg@10']; b=m['als_only']['ndcg@10']; c=m['popularity']['ndcg@10']; assert a>b>c, (a,b,c); print('ordering ok', a, b, c)"`
Expected: `ordering ok ...`.

- [ ] **Step 4: Commit Phase 3**

```bash
git add eval/ train/ranker.py train/baselines.py tests/test_metrics.py artifacts/metrics.json
git commit --author="Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>" -m "Phase 3: LambdaRank ranker, baselines, offline eval + metrics.json"
```

---

## Phase 4: FastAPI Inference Service

**Verification gate:** all 5 endpoints work; `/recommend` < 200 ms; `test_api.py` passes.

### Task 4.1: Pydantic schemas

**Files:**
- Create: `api/schemas.py`

- [ ] **Step 1: Implement `api/schemas.py`**

```python
from pydantic import BaseModel

class Recommendation(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float
    retrieval_score: float

class RecommendResponse(BaseModel):
    user_id: int
    stage: str
    recommendations: list[Recommendation]

class UserProfile(BaseModel):
    user_id: int
    n_ratings: int
    avg_rating: float
    top_genres: list[str]
    genre_affinity: dict[str, float]

class MovieInfo(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    avg_rating: float
    n_ratings: int
    release_year: int | None

class Health(BaseModel):
    status: str
    models_loaded: bool
```

### Task 4.2: Recommender (loads artifacts, two-stage inference)

**Files:**
- Create: `api/recommender.py`

- [ ] **Step 1: Implement `api/recommender.py`**

```python
"""Loads all artifacts once; serves two-stage recommendations."""
import json, numpy as np, joblib, faiss, lightgbm as lgb, polars as pl
from config import MODELS, FEATURES, METRICS_JSON, N_CANDIDATES, GENRES
from eval.featbuild import build_matrix

class Recommender:
    def __init__(self):
        self.als = joblib.load(MODELS / "als.pkl")
        self.index = faiss.read_index(str(MODELS / "faiss.index"))
        self.ranker = lgb.Booster(model_file=str(MODELS / "ranker.txt"))
        self.idx_to_movieid = np.load(MODELS / "idx_to_movieid.npy")
        idx_to_userid = np.load(MODELS / "idx_to_userid.npy")
        self.uidx = {int(u): i for i, u in enumerate(idx_to_userid.tolist())}
        self.items = pl.read_parquet(FEATURES / "items.parquet")
        self.users = pl.read_parquet(FEATURES / "users.parquet")
        self.user_dict = {int(r["userId"]): r for r in self.users.iter_rows(named=True)}
        self.item_dict = {int(r["movieId"]): r for r in self.items.iter_rows(named=True)}
        self.max_item_ratings = int(self.items["n_ratings"].max())
        self.metrics = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
        self.loaded = True

    def known_user(self, uid): return uid in self.uidx

    def _format(self, movieids, scores, retr):
        out = []
        for m, s, r in zip(movieids, scores, retr):
            it = self.item_dict.get(int(m))
            if not it:
                continue
            out.append({"movie_id": int(m), "title": it["title"],
                        "genres": [g for g in it["genres"].split("|")],
                        "score": float(s), "retrieval_score": float(r)})
        return out

    def recommend(self, uid: int, k: int = 10, stage: str = "both"):
        uvec = self.als.user_factors[self.uidx[uid]].astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(uvec)
        D, I = self.index.search(uvec, N_CANDIDATES)
        cands = self.idx_to_movieid[I[0]]
        retr = D[0]
        if stage == "retrieval":
            return self._format(cands[:k], retr[:k], retr[:k])
        X, mids = build_matrix(self.user_dict[uid], cands, self.items, self.max_item_ratings)
        scores = self.ranker.predict(X)
        order = np.argsort(scores)[::-1][:k]
        retr_map = {int(m): float(r) for m, r in zip(cands, retr)}
        chosen = [mids[i] for i in order]
        return self._format(chosen, [scores[i] for i in order],
                            [retr_map[int(m)] for m in chosen])

    def user_profile(self, uid: int):
        r = self.user_dict[uid]
        aff = {g: float(r[f"genre_affinity_{g}"]) for g in GENRES}
        top = sorted(aff, key=aff.get, reverse=True)[:5]
        return {"user_id": uid, "n_ratings": int(r["n_ratings"]),
                "avg_rating": float(r["avg_rating"]), "top_genres": top, "genre_affinity": aff}

    def movie_info(self, mid: int):
        r = self.item_dict[mid]
        yr = r.get("release_year")
        return {"movie_id": mid, "title": r["title"], "genres": r["genres"].split("|"),
                "avg_rating": float(r["avg_rating"]), "n_ratings": int(r["n_ratings"]),
                "release_year": int(yr) if yr is not None else None}
```

### Task 4.3: FastAPI app

**Files:**
- Create: `api/main.py`

- [ ] **Step 1: Implement `api/main.py`**

```python
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from api.recommender import Recommender
from api import schemas

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rec"] = Recommender()
    yield
    state.clear()

app = FastAPI(title="movie-recsys", lifespan=lifespan)

@app.get("/health", response_model=schemas.Health)
def health():
    return {"status": "ok", "models_loaded": bool(state.get("rec"))}

@app.get("/recommend/{user_id}", response_model=schemas.RecommendResponse)
def recommend(user_id: int, k: int = Query(10, ge=1, le=100),
              stage: str = Query("both", pattern="^(both|retrieval|ranker)$")):
    rec = state["rec"]
    if not rec.known_user(user_id):
        raise HTTPException(404, f"user {user_id} not found")
    s = "retrieval" if stage == "retrieval" else "both"
    return {"user_id": user_id, "stage": stage, "recommendations": rec.recommend(user_id, k, s)}

@app.get("/users/{user_id}", response_model=schemas.UserProfile)
def user(user_id: int):
    rec = state["rec"]
    if user_id not in rec.user_dict:
        raise HTTPException(404, f"user {user_id} not found")
    return rec.user_profile(user_id)

@app.get("/movies/{movie_id}", response_model=schemas.MovieInfo)
def movie(movie_id: int):
    rec = state["rec"]
    if movie_id not in rec.item_dict:
        raise HTTPException(404, f"movie {movie_id} not found")
    return rec.movie_info(movie_id)

@app.get("/metrics")
def metrics():
    return state["rec"].metrics
```

### Task 4.4: API smoke tests + timing

**Files:**
- Create: `tests/test_api.py`

- [ ] **Step 1: Write `tests/test_api.py`**

```python
import numpy as np
from fastapi.testclient import TestClient
from api.main import app
from config import MODELS

def _known_user():
    return int(np.load(MODELS / "idx_to_userid.npy")[0])

def test_health():
    with TestClient(app) as c:
        r = c.get("/health"); assert r.status_code == 200
        assert r.json()["models_loaded"] is True

def test_recommend_both():
    with TestClient(app) as c:
        r = c.get(f"/recommend/{_known_user()}?k=10")
        assert r.status_code == 200
        assert len(r.json()["recommendations"]) == 10

def test_recommend_unknown_user():
    with TestClient(app) as c:
        assert c.get("/recommend/999999999").status_code == 404

def test_stage_retrieval_differs():
    with TestClient(app) as c:
        u = _known_user()
        both = [x["movie_id"] for x in c.get(f"/recommend/{u}?stage=both").json()["recommendations"]]
        retr = [x["movie_id"] for x in c.get(f"/recommend/{u}?stage=retrieval").json()["recommendations"]]
        assert both != retr  # re-ranking changed order
```

- [ ] **Step 2: Run API tests**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (4 tests). Loading artifacts in the fixture may take a few seconds.

- [ ] **Step 3: Manual timing check**

Run: `uv run uvicorn api.main:app --port 8000 &` then `sleep 8 && curl -s -w "\n%{time_total}s\n" "http://localhost:8000/recommend/$(uv run python -c "import numpy as np;print(int(np.load('artifacts/models/idx_to_userid.npy')[0]))")?k=10" | tail -1 && kill %1`
Expected: time_total < 0.200s after warmup. (First call may be slower; run twice.)

- [ ] **Step 4: Commit Phase 4**

```bash
git add api/ tests/test_api.py
git commit --author="Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>" -m "Phase 4: FastAPI two-stage inference service + smoke tests"
```

---

## Phase 5: Vanilla-JS Dashboard

**Verification gate:** profile loads, recs render, stage toggle works, metrics panel populated, errors clean. Verified in browser.

### Task 5.1: HTML shell

**Files:**
- Create: `ui/index.html`

- [ ] **Step 1: Write `ui/index.html`** — 3-panel CSS-grid layout: left `#profile` (user-id input + search button + profile output), center `#recs` (stage toggle `both|retrieval` + list), right `#metrics` (model-comparison bars + feature-importance bars). Link `style.css`, load `app.js` with `defer`. All API calls go to `/api/...` (nginx proxies). Include a `<div class="error">` region per panel.

### Task 5.2: Styling (dark theme, monospace scores)

**Files:**
- Create: `ui/style.css`

- [ ] **Step 1: Write `ui/style.css`** — dark bg (`#0d1117`), card panels (`#161b22`), accent for scores, monospace font for numbers, CSS-grid 3 columns (`280px 1fr 360px`), `.bar` class with `width:%` driven inline. Match the rag-assistant aesthetic. Loading spinner + `.error` styles.

### Task 5.3: App logic

**Files:**
- Create: `ui/app.js`

- [ ] **Step 1: Write `ui/app.js`**

Functions:
- `loadUser(id)` → `GET /api/users/{id}` → render profile + genre-affinity mini bar chart (CSS bars from `genre_affinity`). On 404 show "User not found".
- `loadRecs(id, stage)` → `GET /api/recommend/{id}?k=10&stage={stage}` → render numbered list (title · genres · year · score). Store both `both` and `retrieval` results; when toggled, diff positions and show ▲/▼ vs the other list.
- `loadMetrics()` → `GET /api/metrics` once on load → render NDCG@10 comparison bars (popularity/als_only/als_lgbm) + top-10 feature-importance horizontal bars.
- Wire search button + Enter key; show spinners during fetch; catch network errors into the panel `.error` div.

- [ ] **Step 2: Verify with the running API**

Run: serve UI locally against the live API — `uv run uvicorn api.main:app --port 8000 &` then `cd ui && uv run python -m http.server 8080 &`. Temporarily point fetch base to `http://localhost:8000` (or add a `<base>`/const `API`). Open `http://localhost:8080`.

Use Claude Preview (mcp__Claude_Preview__preview_start on `http://localhost:8080`) to:
- enter a known user id (from `idx_to_userid.npy[0]`) → profile renders
- recs list shows 10 items with scores
- toggle Stage 1 / Both → order changes, arrows appear
- metrics bars match `artifacts/metrics.json`
- enter `999999999` → clean "User not found"

Expected: all five behaviors confirmed via screenshot.

- [ ] **Step 3: Commit Phase 5**

```bash
git add ui/index.html ui/style.css ui/app.js
git commit --author="Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>" -m "Phase 5: three-panel vanilla-JS inspection dashboard"
```

---

## Phase 6: Docker Compose + Docs

**Verification gate:** `docker compose up --build` serves API (8000) + UI (80) end-to-end (real image build).

### Task 6.1: Export requirements + Dockerfile

**Files:**
- Create: `requirements.txt`, `Dockerfile.api`

- [ ] **Step 1: Export pinned requirements**

Run: `uv export --no-dev --no-hashes -o requirements.txt`
Expected: `requirements.txt` with pinned versions.

- [ ] **Step 2: Write `Dockerfile.api`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.py .
COPY eval/ ./eval/
COPY api/ ./api/
COPY artifacts/features/ ./artifacts/features/
COPY artifacts/models/ ./artifacts/models/
COPY artifacts/metrics.json ./artifacts/
ENV PYTHONPATH=/app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> `libgomp1` is required by LightGBM at runtime in slim images.

### Task 6.2: nginx config + compose

**Files:**
- Create: `ui/nginx.conf`, `docker-compose.yml`

- [ ] **Step 1: Write `ui/nginx.conf`**

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;
    location /api/ {
        proxy_pass http://api:8000/;
        proxy_set_header Host $host;
    }
    location / { try_files $uri $uri/ /index.html; }
}
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  api:
    build: { context: ., dockerfile: Dockerfile.api }
    ports: ["8000:8000"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5
  ui:
    image: nginx:alpine
    ports: ["80:80"]
    volumes:
      - ./ui:/usr/share/nginx/html:ro
      - ./ui/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      api:
        condition: service_healthy
```

- [ ] **Step 3: Build + run + verify end-to-end**

Run: `docker compose up --build -d` then `sleep 30 && curl -s http://localhost:8000/health && curl -s -o /dev/null -w "ui:%{http_code}\n" http://localhost:80 && curl -s -o /dev/null -w "proxy:%{http_code}\n" http://localhost:80/api/health`
Expected: `{"status":"ok",...}`, `ui:200`, `proxy:200`. Then `docker compose down`.

> If the image build OOMs on 8 GB, build with `DOCKER_BUILDKIT=1` and close other apps; the install step is the heavy part.

### Task 6.3: README + RESULTS

**Files:**
- Create: `README.md`, `RESULTS.md`, `Makefile`

- [ ] **Step 1: Write `Makefile`** with targets: `setup` (uv sync), `data`, `pipeline` (split→genome→items→users), `train` (retrieval→ranker→baselines), `eval`, `api`, `test`, `up`, `down`.

- [ ] **Step 2: Write `RESULTS.md`** — paste the real metrics table from `artifacts/metrics.json` (popularity / ALS / ALS+LambdaRank across NDCG@5/10, Recall@10, Precision@10, MAP@10, Coverage@10, Personalization@10), top-10 feature importance, and the trade-offs section (temporal split, confidence weighting, LambdaRank vs pointwise, IndexFlatIP at 27k, ranker user-sampling for 8 GB, cold-start exclusion).

- [ ] **Step 3: Write `README.md`** — architecture diagram (ASCII), quick start (`docker compose up --build`), train-from-scratch sequence (`make data pipeline train eval`), endpoint table, link to RESULTS.md + the study-guide docx, design decisions.

- [ ] **Step 4: Create GitHub repo + push**

```bash
gh repo create movie-recsys --public --source=. --remote=origin \
  --description "Two-stage movie recommender: ALS+Faiss retrieval, LightGBM LambdaRank re-rank, FastAPI + dashboard"
git add Makefile README.md RESULTS.md requirements.txt Dockerfile.api docker-compose.yml ui/nginx.conf
git commit --author="Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>" -m "Phase 6: Docker Compose, nginx, README + RESULTS"
git push -u origin main
```

---

## Phase 7: Study Guide (docx) — finalize

> Sections are drafted incrementally as each phase completes (per spec). This phase polishes and exports the full document.

### Task 7.1: Technical deep-dive docx

**Files:**
- Create: `output/doc/movie-recsys-technical-deep-dive.docx`, `scripts/build_study_guide.py`

- [ ] **Step 1: Read the reference doc for tone/structure**

Read `~/Desktop/Proyectos/rag-assistant/output/doc/rag-assistant-technical-deep-dive.docx` (use the docx skill) to match heading hierarchy, depth, and voice.

- [ ] **Step 2: Write `scripts/build_study_guide.py`** using `python-docx` (or the docx skill) with sections: (1) Problem & dataset, (2) Why two-stage retrieval+ranking, (3) Temporal split & leakage, (4) Feature engineering walkthrough, (5) ALS math + confidence weighting, (6) Faiss ANN & why IndexFlatIP, (7) LambdaRank & NDCG optimization, (8) Offline metrics explained (NDCG/MAP/coverage/personalization), (9) Real results table from metrics.json, (10) Serving architecture, (11) 8 GB engineering trade-offs, (12) What I'd do in production. Embed the metrics table from `artifacts/metrics.json`.

- [ ] **Step 3: Generate + verify**

Run: `uv run python scripts/build_study_guide.py`
Expected: `output/doc/movie-recsys-technical-deep-dive.docx` opens, has a TOC/headings, and the results table matches metrics.json.

- [ ] **Step 4: Commit + push**

```bash
git add output/doc/movie-recsys-technical-deep-dive.docx scripts/build_study_guide.py
git commit --author="Javier Sánchez Esquivel <jsanchez.ieu2022@student.ie.edu>" -m "Study guide: technical deep-dive docx"
git push
```

---

## Self-Review Notes (spec coverage)

- Temporal split / leakage → Task 1.3 + leakage assertion 1.6.3 ✓
- Genome streaming top-20 (8 GB) → Task 1.4 ✓
- User/item features incl. Bayesian avg, genre affinity → 1.5, 1.6 ✓
- ALS + Faiss, Recall@200 gate → Task 2.1 ✓
- Ranker user-sampling (40k) for 8 GB → Task 3.3 ✓
- Shared feature builder (no train/serve skew) → Task 3.2, reused in 3.3/3.5/4.2 ✓
- 3-way eval + coverage + personalization → Task 3.5 ✓
- metrics.json committed → 3.5.4 ✓
- 5 API endpoints + <200ms → Phase 4 ✓
- 3-panel dashboard + stage toggle → Phase 5 ✓
- Full Docker build+verify → Task 6.2.3 ✓
- README/RESULTS/trade-offs → Task 6.3 ✓
- Incremental docx study guide → Phase 7 ✓
- Public repo, author w/o co-author trailer → 6.3.4 + every commit ✓
```
