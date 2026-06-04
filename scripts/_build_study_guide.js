// Authoring source for the movie-recsys technical deep-dive study guide.
// Run via `python scripts/build_study_guide.py` (resolves the global `docx`
// package and post-processes heading styles). Output path is argv[2].
//
// Style mirrors the rag-assistant reference deep-dive: a title page with a
// subtitle and repo/stack line, a Table of Contents, then numbered Heading-1
// sections with Heading-2 subsections, taught in substantive prose.

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, Footer,
} = require("docx");

const OUT = process.argv[2] || "movie-recsys-technical-deep-dive.docx";

// ---------- helpers -------------------------------------------------------

const FONT = "Arial";
const CONTENT_W = 9360; // US Letter, 1" margins

// Parse inline `code` spans (backticks) and **bold** spans into TextRuns.
function runs(text) {
  const out = [];
  const parts = text.split(/(`[^`]+`)/g);
  for (const part of parts) {
    if (part.startsWith("`") && part.endsWith("`")) {
      out.push(new TextRun({ text: part.slice(1, -1), font: "Consolas", size: 20 }));
    } else {
      const bolds = part.split(/(\*\*[^*]+\*\*)/g);
      for (const b of bolds) {
        if (b.startsWith("**") && b.endsWith("**")) {
          out.push(new TextRun({ text: b.slice(2, -2), bold: true }));
        } else if (b) {
          out.push(new TextRun(b));
        }
      }
    }
  }
  return out.length ? out : [new TextRun(text)];
}

function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 140, line: 276 }, children: runs(text), ...opts });
}
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 90, line: 276 },
    children: runs(text),
  });
}
function tocLine(text) {
  return new Paragraph({ spacing: { after: 40 }, children: runs(text) });
}
function spacer() {
  return new Paragraph({ spacing: { after: 80 }, children: [new TextRun("")] });
}

// Bordered table from a header row + body rows (arrays of strings).
function table(header, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mk = (text, w, opts = {}) =>
    new TableCell({
      borders,
      width: { size: w, type: WidthType.DXA },
      margins: { top: 60, bottom: 60, left: 110, right: 110 },
      shading: opts.shade ? { fill: opts.shade, type: ShadingType.CLEAR } : undefined,
      children: [new Paragraph({
        spacing: { after: 0, line: 252 },
        children: [new TextRun({ text, bold: !!opts.bold, size: 19 })],
      })],
    });
  const headRow = new TableRow({
    tableHeader: true,
    children: header.map((t, i) => mk(t, widths[i], { bold: true, shade: "D9E2F0" })),
  });
  const bodyRows = rows.map((r) =>
    new TableRow({
      children: r.map((t, i) => mk(t, widths[i], { bold: i === 0 && r._bold, shade: r._shade })),
    }));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headRow, ...bodyRows],
  });
}

// Shaded single-cell callout for a key takeaway.
function callout(text) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "C9C9C9" };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: { top: border, bottom: border, left: border, right: border },
      width: { size: CONTENT_W, type: WidthType.DXA },
      margins: { top: 100, bottom: 100, left: 140, right: 140 },
      shading: { fill: "F1F4F9", type: ShadingType.CLEAR },
      children: [new Paragraph({ spacing: { after: 0, line: 276 }, children: runs(text) })],
    })] })],
  });
}

// ---------- content -------------------------------------------------------

const children = [];

// Title block
children.push(new Paragraph({
  spacing: { before: 600, after: 120 },
  children: [new TextRun({ text: "movie-recsys — Technical Deep-Dive", bold: true, size: 48, font: FONT })],
}));
children.push(new Paragraph({
  spacing: { after: 200 },
  children: [new TextRun({ text: "A from-scratch study guide to a two-stage movie recommender: ALS retrieval, LightGBM LambdaRank re-ranking, honest temporal evaluation, and a served inference API.", italics: true, size: 24, color: "555555" })],
}));
children.push(new Paragraph({
  spacing: { after: 60 },
  children: [
    new TextRun({ text: "Repository: javsanesq/movie-recsys", size: 20, color: "555555" }),
    new TextRun({ text: "   |   Dataset: MovieLens 25M", size: 20, color: "555555" }),
    new TextRun({ text: "   |   Stack: Polars · implicit (ALS) · LightGBM · FastAPI · Docker", size: 20, color: "555555" }),
  ],
}));
children.push(new Paragraph({
  spacing: { after: 200 },
  children: [new TextRun({ text: "Audience: the author, revisiting the system end-to-end and preparing to explain it. Every section explains not just what the code does, but why it was built that way.", size: 20, color: "555555" })],
}));

// TOC
children.push(h1("Table of Contents"));
[
  "1. Problem and dataset",
  "2. Two-stage architecture",
  "3. Temporal split and leakage",
  "4. Feature engineering",
  "5. Stage 1 — ALS retrieval",
  "6. Stage 2 — LightGBM LambdaRank",
  "7. Offline evaluation",
  "8. Results",
  "9. Serving architecture",
  "10. Engineering under an 8 GB budget",
  "11. What I'd do in production",
].forEach((t) => children.push(tocLine(t)));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 1. Problem & dataset ==============================================
children.push(h1("1. Problem and dataset"));
children.push(p("Recommendation is best framed not as predicting a star rating but as **ranking**: given a user, produce an ordered shortlist of items they are most likely to engage with next. The user only ever sees the top of the list, so what matters is whether the right items land near the top — not whether a regression model nails the numeric rating of every movie in the catalog. This reframing is the single most important mental shift in the project, and it dictates the models, the labels, and the metrics that follow."));
children.push(p("The data is **MovieLens 25M**, a standard public benchmark from GroupLens. It ships as a handful of CSVs:"));
children.push(bullet("`ratings.csv` — about **25 million rows** of `(userId, movieId, rating, timestamp)`. Ratings are on a 0.5–5.0 scale in half-star steps. The timestamp is a Unix epoch, and it is the backbone of the whole project: it lets us order events in time and split honestly."));
children.push(bullet("`movies.csv` — `(movieId, title, genres)`. The title embeds the release year in parentheses, e.g. `Toy Story (1995)`; genres are **pipe-delimited**, e.g. `Adventure|Animation|Children|Comedy`."));
children.push(bullet("`genome-scores.csv` — roughly **30 million rows** of `(movieId, tagId, relevance)`, a dense soft-tagging of each movie against ~1,100 tags (a relevance score in [0,1] for every movie–tag pair). This is the richest content signal available, and also the largest file."));
children.push(bullet("`genome-tags.csv` — the `(tagId, tag)` lookup that turns a numeric tagId into a human-readable tag name."));
children.push(p("Two conventions matter when reading the code. First, `userId` and `movieId` are **1-indexed external IDs**, not array positions — the models work in dense internal indices and we carry explicit `idx_to_movieid` / `idx_to_userid` maps to translate back and forth. Second, genres are a delimited string, so genre features are built with substring containment (`genres.str.contains(\"Comedy\")`) rather than a parsed list."));
children.push(callout("**Why this dataset.** 25M is large enough to be realistic — memory and time genuinely become constraints, which forces real engineering — yet small enough to train end-to-end on a laptop. The genome tags give a strong content signal that lets the re-ranker do more than memorize popularity."));

// ===== 2. Two-stage architecture =========================================
children.push(h1("2. Two-stage architecture"));
children.push(p("The system uses the **retrieve-then-rank** pattern that underpins essentially every large-scale industrial recommender (YouTube, Netflix, Pinterest, and so on). The catalog has ~18,000 eligible items; scoring all of them per user with an expensive model would be wasteful, so the work is split into two stages with very different jobs."));
children.push(h2("2.1 Stage 1 — candidate generation (retrieval)"));
children.push(p("A cheap model — here **ALS matrix factorization** — scans the entire catalog and pulls back a small candidate set (the top **200** items per user). Its only job is **recall**: get as many of the user's truly-relevant items as possible into those 200, as fast as possible. It does not need to order them well. Casting a wide, diverse net is the whole point."));
children.push(h2("2.2 Stage 2 — ranking"));
children.push(p("An expensive, feature-rich model — here a **LightGBM LambdaRank** learner — sees only the 200 candidates and re-orders them using dozens of user, item, and cross features. Its job is **precision at the top**: make sure the best handful of the 200 surface first. Because it only ever scores 200 items instead of 18,000, we can afford a model that would be far too slow to run over the full catalog."));
children.push(callout("**The core idea:** Stage 1 maximizes recall cheaply over the whole catalog; Stage 2 maximizes precision expensively over a tiny candidate set. Neither stage can do the other's job well — and Section 8 shows this empirically: ALS-only actually ranks below a popularity baseline, yet adding the re-ranker on top of those same ALS candidates beats everything."));
children.push(p("This division also shapes how the two models are evaluated and tuned. Stage 1 is judged by Recall@200 (did we capture the relevant items at all?); Stage 2 is judged by NDCG, Precision, and MAP at small k (did we order the survivors well?)."));

// ===== 3. Temporal split & leakage =======================================
children.push(h1("3. Temporal split and leakage"));
children.push(p("Every record carries a timestamp, and the data is split strictly in time (`pipeline/split.py`):"));
children.push(bullet("**Train:** all ratings before `2017-01-01`."));
children.push(bullet("**Validation:** ratings during `2017`."));
children.push(bullet("**Test:** ratings from `2018-01-01` onward."));
children.push(p("The boundaries live in `config.py` as Unix epochs (`TRAIN_END`, `VAL_END`), and the split is done with a streaming Polars scan so the 25M-row file never has to fit in memory at once."));
children.push(h2("3.1 Why not a random split"));
children.push(p("A random 80/20 split silently **leaks the future**. If a user rated a movie in 2018 and that rating lands in the training set, the model learns from information that would not have existed at the moment we are pretending to recommend. The offline metric then looks great and collapses in production. A temporal split is the only honest evaluation of a system that, by definition, recommends the future from the past. Concretely: a movie a user rates in 2018 must never shape recommendations evaluated as of 2016."));
children.push(p("This discipline propagates everywhere. **All user and item features are computed on the train period only** — average ratings, genre affinities, popularity counts, Bayesian item means. A feature that peeked at 2017–2018 activity would reintroduce exactly the leak the split was designed to remove."));
children.push(h2("3.2 Warm-start filtering and cold-start"));
children.push(p("After splitting, users with fewer than **5** train ratings and items with fewer than **10** train ratings are dropped (`MIN_USER_TRAIN_RATINGS`, `MIN_ITEM_TRAIN_RATINGS`). Matrix factorization needs a few interactions per entity to estimate a meaningful latent vector; below that threshold the vector is mostly noise. This makes the project an honest **warm-start** system: it serves users and items it has actually seen enough of."));
children.push(callout("**Cold-start is excluded, not hidden.** Brand-new users and items fall below the support thresholds and are removed from training and evaluation. This is documented rather than papered over — and Section 11 describes the popularity + content-based fallback a production system would add for them."));

// ===== 4. Feature engineering ============================================
children.push(h1("4. Feature engineering"));
children.push(p("Stage 2 lives or dies on its features. The project builds **73 features** in total, grouped into per-user, per-item, and cross features. All are computed on the train split only (Section 3) and assembled by a single shared builder (`eval/featbuild.py`) so training, evaluation, and serving see identical inputs."));
children.push(h2("4.1 Per-user features (pipeline/user_features.py)"));
children.push(p("Computed by grouping the train ratings per user:"));
children.push(bullet("`n_ratings` — how many movies the user rated (an activity / experience signal)."));
children.push(bullet("`avg_rating`, `std_rating` — the user's mean rating and its spread. Some users are generous, some harsh; the spread captures how discriminating they are."));
children.push(bullet("`days_since_last` — days between the user's most recent train rating and the train cutoff, a **recency** signal."));
children.push(bullet("`pct_high_ratings` — the fraction of the user's ratings that are ≥ 4.0, i.e. how often they actually love something."));
children.push(bullet("**20 genre-affinity fractions** — for each genre, the fraction of the user's rated movies that carry it. This is the user's taste profile as a 20-dimensional vector, and it powers the most important cross-feature below."));
children.push(h2("4.2 Per-item features (pipeline/item_features.py)"));
children.push(bullet("`release_year` — extracted from the title with a regex (`\\((\\d{4})\\)`)."));
children.push(bullet("`n_ratings`, `std_rating` — item popularity and rating spread."));
children.push(bullet("**20 binary genre flags** — one 0/1 column per genre."));
children.push(bullet("**Top-20 genome tags** — the 20 highest-mean-relevance tags across the catalog, joined per movie (see 4.4)."));
children.push(bullet("**Bayesian average rating** — the headline item feature, explained next."));
children.push(h2("4.3 Bayesian average rating (shrinkage)"));
children.push(p("A naive item mean is dangerous for low-count items: a movie with a single 5.0 rating would appear to be the best film ever made. The fix is **shrinkage toward the global mean**. With `C` = prior count (here **25**, the `BAYES_MIN_COUNT` config) and `m` = the global mean rating across all train ratings, the item's score is:"));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 60 },
  children: [new TextRun({ text: "avg_rating  =  (n · local_mean  +  C · m) / (n + C)", font: "Consolas", size: 22, bold: true })],
}));
children.push(p("Intuitively, every item starts with **25 phantom ratings pinned at the global mean**. An item with thousands of ratings is barely moved — its own data dominates. An item with two ratings is pulled almost all the way back to the global average, because two data points are not evidence of greatness. As real ratings accumulate, the prior's influence fades smoothly. This is why shrinkage matters: it stops the long tail of sparsely-rated movies from polluting the candidate ordering with statistical noise, and it gives the ranker a trustworthy quality signal (`avg_rating` is the 3rd most important feature — Section 8)."));
children.push(h2("4.4 Top-20 genome tags (streamed before pivoting)"));
children.push(p("The genome file is ~30M rows — pivoting it naively into a wide (movie × 1,100-tag) matrix would blow the 8 GB budget instantly. The trick (`pipeline/build_genome_pivot.py`) is to **select and filter before pivoting**: a streaming scan computes each tag's mean relevance, keeps only the **top 20** tags, filters the 30M rows down to just those tags, and *then* pivots. The wide matrix that finally materializes is 20 columns, not 1,100. Order of operations is the whole game here."));
children.push(h2("4.5 Cross-features (eval/featbuild.py)"));
children.push(p("Four features combine user and item signals — these are where personalization actually happens, because they cannot be computed from either side alone:"));
children.push(bullet("`genre_match` — the **dot product of the user's 20-d genre-affinity vector with the candidate's 20-d binary genre vector**. High when a movie's genres align with what the user habitually rates. This is the 2nd most important feature in the whole model."));
children.push(bullet("`rating_delta` — the item's Bayesian mean minus the user's mean rating, normalizing item quality to the user's personal scale."));
children.push(bullet("`popularity_rank` — the item's rating count divided by the catalog max, a normalized popularity in [0,1]."));
children.push(bullet("`user_item_era_diff` — the absolute gap between the movie's release year and the user's approximate active era (derived from `days_since_last`), capturing whether someone reaches for new releases or old favorites."));
children.push(callout("**73 features:** 44 item columns (4 numeric + 20 genre flags + 20 genome tags) + 25 per-user columns (5 numeric + 20 genre affinities) + 4 cross-features. The exact column order is defined once in `feature_columns()` and reused everywhere, which is what guarantees no train/serve skew."));

// ===== 5. Stage 1 — ALS ==================================================
children.push(h1("5. Stage 1 — ALS retrieval"));
children.push(p("Stage 1 is **Alternating Least Squares** matrix factorization on implicit feedback, via the `implicit` library (`train/retrieval.py`)."));
children.push(h2("5.1 Implicit feedback and confidence weighting"));
children.push(p("Explicit ratings are reinterpreted as **implicit signals of engagement**: the act of rating a movie is itself a positive interaction. Rather than treating a 1-star and a 5-star identically, each interaction gets a confidence weight:"));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 60 },
  children: [new TextRun({ text: "confidence  =  1 + α · rating       (α = 10)", font: "Consolas", size: 22, bold: true })],
}));
children.push(p("A 5-star interaction carries far more weight (51) than a 1-star one (11), so the model trusts strong signals more — but crucially, **absence is never treated as a hard negative**. An unrated movie simply has no confidence-weighted entry; it is missing, not disliked. That is the defining property of implicit-feedback ALS and the reason it suits this data."));
children.push(h2("5.2 Training and the latent space"));
children.push(p("`AlternatingLeastSquares(factors=128, iterations=20, regularization=0.01)` learns a **128-dimensional latent vector for every user and every item**. ALS alternates: hold the item vectors fixed and solve for the user vectors in closed form, then swap — each half is a least-squares problem, hence the name. The result is two matrices, `user_factors` and `item_factors`, whose inner products approximate the confidence matrix."));
children.push(h2("5.3 Retrieval = exact top-k inner product"));
children.push(p("To get candidates for a user we take their latent vector and compute its inner product against every item vector — `scores = item_vectors @ user_vec` — then `argpartition` for the top 200. Two deliberate choices drive quality:"));
children.push(bullet("**Raw dot product, not cosine.** L2-normalizing would optimize cosine similarity, but ALS was *trained* on the raw dot product. Searching with cosine would mean scoring with a different objective than the model was fit on. Matching the training objective is both correct and worth about a point of recall."));
children.push(bullet("**Exclude already-seen items.** Each user's train-rated movies are filtered out of the candidate set (the retriever over-fetches by the number of seen items so 200 *novel* candidates survive). Recommending a movie the user already rated wastes a slot; removing them frees ~8 points of recall."));
children.push(p("Together these lift **Recall@200 from 0.187 (naive) to 0.297** on the validation set — meaning roughly 30% of each user's genuinely-future positive items make it into the 200 candidates. On a strict temporal split where the targets are *future* ratings, ~0.30 is the honest, healthy number; the original 0.50 target was an unvalidated guess and is reported as-is rather than reverse-engineered."));
children.push(h2("5.4 Why brute-force numpy instead of Faiss"));
children.push(p("An approximate-nearest-neighbor index (Faiss) is the textbook tool for retrieval — but here it was **deliberately removed** for two reasons:"));
children.push(bullet("**It adds no value at this scale.** Exact `item_vectors @ user_vec` over ~18k items takes ~2 ms. An ANN index trades exactness for speed; at 18k items there is no speed to gain and only accuracy to lose."));
children.push(bullet("**It causes a fatal OpenMP conflict.** `faiss-cpu` and `lightgbm` each *statically* link their own copy of the OpenMP runtime (`libomp`). When both are imported in one process the duplicate runtimes double-initialize and **deadlock** — and both the evaluation path and the serving path import both libraries. Exact numpy search sidesteps the entire problem. This is the kind of integration bug that only surfaces when two ML libraries meet in one process, and it is worth understanding deeply."));

// ===== 6. Stage 2 — LambdaRank ===========================================
children.push(h1("6. Stage 2 — LightGBM LambdaRank"));
children.push(p("Stage 2 is a gradient-boosted decision-tree model trained with the **LambdaRank** objective (`train/ranker.py`)."));
children.push(h2("6.1 Why learning-to-rank, not regression"));
children.push(p("A pointwise model (predict each rating with MSE, then sort) optimizes the wrong thing: it spends capacity getting the rating of item #150 right while ignoring whether the top 5 are ordered well. **LambdaRank optimizes NDCG directly** — it weights each pair-swap by how much swapping those two items would change NDCG, so the gradient focuses learning on the ordering near the top of the list, which is the only part a user sees. For a ranking problem, optimizing the ranking metric beats optimizing a proxy like squared error."));
children.push(h2("6.2 Labels and training candidates"));
children.push(p("Training data is built from the **ALS top-200 candidates per user**. The label for each candidate is `min(4, floor(train_rating))` if the user rated it, else `0` — a graded relevance from 0 (irrelevant) to 4 (loved). Two subtleties:"));
children.push(bullet("**Training candidates are NOT seen-filtered.** Evaluation excludes already-rated items, but training deliberately keeps them — they are the *only* source of positive labels. If we filtered them out, almost every candidate would be label 0 and the model would have nothing to rank toward. About **36% of training candidates are positive** as a result."));
children.push(bullet("`label_gain = [0, 1, 3, 7, 15]` — the relevance-to-gain map fed to NDCG. It is convex: moving from label 3 to 4 (7→15) is worth far more than 0 to 1, so the model is rewarded most for surfacing the movies a user truly loved."));
children.push(h2("6.3 The early-stopping trap and its fix"));
children.push(p("This is the subtlest lesson in the project. Early stopping needs a validation signal to decide when to stop adding trees. The intuitive choice — the temporal validation split — is **too sparse**: after seen-filtering, most users have only a handful of positives among 200 candidates, the NDCG signal is almost flat, and the model **early-stops at ~2 trees**, producing a barely-trained ranker."));
children.push(p("The fix: early-stop on a **held-out slice of TRAIN-labeled users** (`VAL_USERS = 3000`), where every candidate carries a dense train-rating label. With a real signal to climb, the model trains to roughly **490 trees** before stopping. The temporal val/test splits are then reserved for the final, unbiased model comparison in `eval/evaluate.py` — they are never touched during fitting."));
children.push(callout("**Key insight:** match your early-stopping signal to the density of your labels, not to whichever split you happen to call “validation.” A sparse validation signal silently strangles a boosted model."));
children.push(h2("6.4 Hyperparameters and the 8 GB cache"));
children.push(p("`num_leaves = 63`, `learning_rate = 0.05`, `min_child_samples = 50`. The ranker trains on a **20,000-user sample** (~4M candidate rows) to fit the memory budget. Building those candidate matrices takes ~14 minutes, so they are **cached to an `.npz` file** — once cached, LightGBM hyperparameters can be re-tuned in seconds without rebuilding candidates from ALS."));

// ===== 7. Offline evaluation =============================================
children.push(h1("7. Offline evaluation"));
children.push(p("Evaluation (`eval/evaluate.py`, `eval/metrics.py`) compares three systems — a popularity baseline, ALS-only, and the full two-stage pipeline — on the held-out **test** split. For each of the **3,646 test users with ≥ 3 future positive items** (rating ≥ 4.0), candidates are retrieved, seen-filtered, scored, and compared against that user's future positives. The metrics, in plain terms:"));
children.push(bullet("**NDCG@k** (Normalized Discounted Cumulative Gain) — the headline metric. It rewards putting relevant items near the top: each hit is discounted by its position (a hit at rank 1 is worth more than at rank 10), then normalized against the best-possible ordering. This is exactly what LambdaRank optimizes."));
children.push(bullet("**Recall@k** — of the user's relevant items, what fraction appear in the top k. Did we find them at all?"));
children.push(bullet("**Precision@k** — of the top k we showed, what fraction are relevant. Is the shortlist clean?"));
children.push(bullet("**MAP@k** (Mean Average Precision) — averages precision at each relevant hit, rewarding both finding relevant items and ranking them early."));
children.push(bullet("**Coverage@10** — the fraction of the *catalog* that appears across all users' top-10 lists. A system that recommends the same 50 blockbusters to everyone has near-zero coverage; high coverage means the catalog's breadth is actually used."));
children.push(bullet("**Personalization@10** — one minus the average overlap between different users' top-10 lists. High personalization means different users get genuinely different lists rather than the same popular set."));
children.push(p("Coverage and personalization are diagnostic, not targets — they explain *behavior*. As Section 8 shows, ALS has very high personalization but mediocre ranking, which is precisely the candidate-generator signature."));

// ===== 8. Results ========================================================
children.push(h1("8. Results"));
children.push(p("The headline comparison on the 3,646-user test set:"));
children.push(table(
  ["Model", "NDCG@5", "NDCG@10", "Recall@10", "Precision@10", "MAP@10"],
  [
    ["Popularity", "0.0804", "0.0724", "0.0285", "0.0614", "0.0355"],
    ["ALS (Stage 1)", "0.0656", "0.0662", "0.0291", "0.0621", "0.0291"],
    Object.assign(["ALS + LambdaRank", "0.1230", "0.1148", "0.0495", "0.0991", "0.0589"], { _bold: true, _shade: "E4ECDA" }),
  ],
  [2160, 1440, 1440, 1440, 1530, 1350],
));
children.push(spacer());
children.push(p("**The two-stage system wins every metric.** NDCG@10 is **+58% over popularity** and **+73% over ALS-only**; Precision@10 and Recall@10 roughly double. That is the whole thesis demonstrated in one table."));
children.push(h2("8.1 Why ALS-only is below popularity (the key finding)"));
children.push(p("The most instructive result is that **ALS-only scores below the trivial popularity baseline on NDCG**. This looks like failure but is exactly the expected behavior, and understanding why is the point of the project. ALS is a **recall-oriented candidate generator**, not a precision ranker. Its diagnostic metrics prove it: **Coverage@10 = 0.22** (vs popularity's 0.015) and **Personalization = 0.996** — it casts an extremely wide, diverse net. But its raw top-10 *ordering* is worse than just showing globally popular movies."));
children.push(p("The **Stage-2 re-ranker is precisely what fixes this**: it reorders the same 200 ALS candidates and beats *both* baselines. Stage 1 supplies recall and diversity; Stage 2 supplies precision. Neither alone is enough — which is the entire argument for a two-stage architecture, validated rather than asserted."));
children.push(h2("8.2 Feature importance"));
children.push(p("LightGBM gain, top 5:"));
children.push(table(
  ["Feature", "Gain share", "What it captures"],
  [
    ["n_ratings", "0.354", "item popularity"],
    ["genre_match", "0.111", "user affinity · item genres (the learned cross-feature)"],
    ["avg_rating", "0.066", "Bayesian item quality"],
    ["days_since_last", "0.063", "user recency"],
    ["user_item_era_diff", "0.040", "release-year vs user-era gap"],
  ],
  [2600, 1500, 5260],
));
children.push(spacer());
children.push(p("Item popularity dominates but does **not monopolize**. The learned `genre_match` cross-feature, the Bayesian quality signal, and per-user recency all contribute meaningfully — which is *why* the re-ranker personalizes instead of collapsing into a popularity list. Just as important: **no single feature approaches a gain share of 1.0**. If one feature did, it would usually mean a target leak — a feature that secretly encodes the label. The healthy spread here is evidence the model is learning a genuine, multi-signal ranking function."));

// ===== 9. Serving ========================================================
children.push(h1("9. Serving architecture"));
children.push(p("The model is served by a **FastAPI** app (`api/main.py`, `api/recommender.py`). All artifacts — the ALS model, item vectors, the LightGBM booster, feature tables, the seen-item map, and the metrics file — are loaded **once at startup** via the lifespan hook and held in memory. A request never touches disk for model weights, so latency is ~**13 ms** per request."));
children.push(h2("9.1 Endpoints"));
children.push(bullet("`GET /recommend/{user_id}?k=10&stage=both|retrieval` — top-k recommendations. `stage=retrieval` returns raw ALS candidates (Stage 1 only); `stage=both` runs the full pipeline and returns the re-ranked list with both a ranker `score` and the underlying `retrieval_score`. The toggle lets you *see* the re-ranker's effect live."));
children.push(bullet("`GET /users/{id}` — profile: n_ratings, avg rating, top genres, and the full genre-affinity vector."));
children.push(bullet("`GET /movies/{id}` — title, genres, Bayesian average rating, release year."));
children.push(bullet("`GET /metrics` — the offline model comparison and feature importance (so the dashboard can chart them)."));
children.push(bullet("`GET /health` — liveness and a `models_loaded` flag."));
children.push(h2("9.2 No train/serve skew"));
children.push(p("This is the architectural keystone. The serving path in `api/recommender.py` calls **the exact same `retrieve()` and `build_matrix()` functions** used by training (`train/ranker.py`) and offline evaluation (`eval/evaluate.py`). Feature definitions, column order, and the seen-filtering rule are defined once and shared, so a movie scored offline is scored identically online. Train/serve skew — features computed one way in training and another in production — is one of the most common and most insidious ML production bugs, and sharing the builder eliminates it by construction."));
children.push(h2("9.3 UI and deployment"));
children.push(p("A **vanilla-JS three-panel dashboard** (no framework) inspects the system: a user profile with genre-affinity bars, live recommendations with the Stage-1/both toggle, and the offline model-comparison and feature-importance charts. **nginx** serves the static UI and same-origin-proxies `/api/*` to FastAPI, which avoids CORS entirely. The whole thing is a two-service **Docker Compose** stack (api + ui) that comes up with one command."));

// ===== 10. 8 GB trade-offs ===============================================
children.push(h1("10. Engineering under an 8 GB budget"));
children.push(p("The project was built to run end-to-end on an 8 GB machine. Every memory compromise is deliberate and documented — and each one is a small lesson in working within real constraints:"));
children.push(bullet("**Streamed genome top-20 before pivot.** Filtering ~30M genome rows down to 20 tags *before* pivoting keeps the wide matrix at 20 columns instead of ~1,100 (Section 4.4)."));
children.push(bullet("**20k-user ranker sample.** The full user base would not fit the candidate matrices in memory; a 20,000-user sample (~4M rows) preserves signal while fitting the budget (Section 6.4)."));
children.push(bullet("**Capped evaluation.** Evaluation runs on a bounded user sample so the comparison is fast and memory-stable."));
children.push(bullet("**numpy over Faiss.** Exact search needs no index in memory and dodges the OpenMP deadlock (Section 5.4)."));
children.push(bullet("**`.npz` candidate cache.** The expensive candidate-build step is cached so the ranker can be re-tuned without rebuilding from ALS (Section 6.4)."));
children.push(p("Streaming Polars throughout (lazy scans, streaming collects) means the 25M-row ratings file and the 30M-row genome file are never fully materialized in memory at once. Working within a constraint, and naming each trade-off explicitly, is itself a portfolio-worthy signal of engineering maturity."));

// ===== 11. Production ====================================================
children.push(h1("11. What I'd do in production"));
children.push(p("This is an honest, complete *offline* system. Scaling it to a live product would add:"));
children.push(bullet("**Cold-start fallback.** Serve new users and items (excluded here by the support thresholds) with a popularity + content-based blend until they accumulate enough interactions for ALS to learn a meaningful latent vector."));
children.push(bullet("**Online / incremental retraining.** Tastes and the catalog drift; the temporal split that protects offline honesty also means a model trained on pre-2017 data goes stale. A production system retrains — or incrementally updates — on a schedule and monitors for drift."));
children.push(bullet("**A/B testing.** Offline NDCG is a proxy for what we actually care about: engagement. Online A/B tests measure the real outcome and catch cases where offline gains do not translate."));
children.push(bullet("**A real ANN index and a feature store at scale.** At 18k items brute-force search is fine, but a catalog of millions needs Faiss / ScaNN (with the OpenMP conflict resolved by process isolation). A feature store would serve precomputed user/item features at low latency and keep training and serving in lockstep."));
children.push(bullet("**Monitoring.** Track latency, recall/precision proxies, coverage and personalization over time, plus data-quality alerts — so silent degradation is caught before users feel it."));
children.push(callout("**The through-line:** every decision in this project — temporal split, two stages, shrinkage, shared feature builder, numpy retrieval — was made to be honest and correct first, and is documented so the reasoning survives. That is the habit this build is meant to demonstrate."));

// ---------- assemble ------------------------------------------------------

const doc = new Document({
  creator: "Javier Sánchez Esquivel",
  title: "movie-recsys — Technical Deep-Dive",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F3864" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "2E5496" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "movie-recsys — Technical Deep-Dive    ", size: 16, color: "888888" }),
          new TextRun({ text: "Page ", size: 16, color: "888888" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" }),
        ],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT);
});
