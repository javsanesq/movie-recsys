// movie-recsys — Technical Deep-Dive study guide generator (docx-js).
// Authored to match the rag-assistant reference deep-dive in tone, heading
// hierarchy, and structure. Run via scripts/build_study_guide.py.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, TableOfContents, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, Header, Footer, PageBreak,
} = require("docx");

const OUT = process.argv[2] || "movie-recsys-technical-deep-dive.docx";

// ---- helpers ---------------------------------------------------------------
const CONTENT_W = 9360; // US Letter, 1" margins

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
// paragraph from an array of [text, {opts}] runs or a plain string
function p(content, opts = {}) {
  let runs;
  if (typeof content === "string") runs = [new TextRun(content)];
  else runs = content.map((r) => (typeof r === "string" ? new TextRun(r) : new TextRun(r)));
  return new Paragraph({ spacing: { after: 140 }, ...opts, children: runs });
}
function bullet(content) {
  const runs = typeof content === "string" ? [new TextRun(content)] : content.map((r) => new TextRun(r));
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: runs });
}
function numItem(content) {
  const runs = typeof content === "string" ? [new TextRun(content)] : content.map((r) => new TextRun(r));
  return new Paragraph({ numbering: { reference: "steps", level: 0 }, spacing: { after: 60 }, children: runs });
}
function code(text) {
  return new Paragraph({
    spacing: { after: 140 },
    shading: { fill: "F2F2F2", type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: "Consolas", size: 19 })],
  });
}
const B = (t) => ({ text: t, bold: true });
const T = (t) => ({ text: t });
const M = (t) => ({ text: t, font: "Consolas", size: 20 }); // monospace inline

// ---- table builder ---------------------------------------------------------
const cellBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };
function cell(text, width, { header = false, bold = false, align } = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold: header || bold, size: 20 })],
    })],
  });
}
function table(widths, rows) {
  const trs = rows.map((cells, ri) =>
    new TableRow({
      tableHeader: ri === 0,
      children: cells.map((c, ci) =>
        cell(c, widths[ci], { header: ri === 0, align: ci === 0 ? AlignmentType.LEFT : AlignmentType.LEFT })),
    }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows: trs });
}

// ===========================================================================
const children = [];

// ---- Title block -----------------------------------------------------------
children.push(new Paragraph({
  spacing: { after: 80 },
  children: [new TextRun({ text: "movie-recsys — Technical Deep-Dive", bold: true, size: 44 })],
}));
children.push(new Paragraph({
  spacing: { after: 80 },
  children: [new TextRun({ text: "A from-scratch study guide to a two-stage movie recommender: retrieve with ALS, re-rank with LightGBM LambdaRank, evaluate honestly on a temporal split, and serve behind a FastAPI inference API.", italics: true, size: 24 })],
}));
children.push(new Paragraph({
  spacing: { after: 240 },
  children: [new TextRun({ text: "Repository: javsanesq/movie-recsys  |  Dataset: MovieLens 25M  |  Stack: Python · implicit (ALS) · LightGBM · FastAPI · Docker", size: 20, color: "555555" })],
}));

// ---- Table of contents -----------------------------------------------------
children.push(h1("Table of Contents"));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===========================================================================
// 1. Problem & dataset
// ===========================================================================
children.push(h1("1. Problem and Dataset"));
children.push(p("Recommendation is, at heart, a ranking problem. Given a user, the system must order the catalog so that the items the user would most enjoy appear at the top of a short list. It is not a rating-prediction problem: nobody scrolls to the bottom of a list, so what matters is the quality of the first five or ten positions, not how accurately we predict a 3.5-star rating on an item the user will never see. Framing the task as ranking — rather than regression — drives every later design choice, from the loss function to the evaluation metrics."));
children.push(p([B("The dataset is MovieLens 25M"), T(", a standard academic benchmark from GroupLens. It is large enough to be realistic and small enough to train on a laptop. The four files this project uses are:")]));
children.push(table([2400, 2200, 4760], [
  ["File", "Approx. size", "Contents"],
  ["ratings.csv", "25M rows", "userId, movieId, rating (0.5–5.0 in 0.5 steps), timestamp. The core interaction log."],
  ["movies.csv", "~62k rows", "movieId, title (with release year in parentheses), pipe-delimited genres."],
  ["genome-scores.csv", "~30M rows", "movieId, tagId, relevance (0–1): a dense machine-learned tag-relevance matrix."],
  ["genome-tags.csv", "~1k rows", "tagId → human-readable tag name (e.g. \"dark\", \"based on a book\")."],
]));
children.push(p("A few dataset conventions matter for correctness. The userId and movieId fields are 1-indexed and not contiguous — there are gaps — so the code never assumes ids equal array positions; it builds explicit id-to-index maps. Genres are a single pipe-delimited string per movie (for example Action|Adventure|Sci-Fi), and the special token (no genres listed) is a real category that must be handled, not dropped. The genome data is the project's richest content signal: for thousands of movies it gives a relevance score against ~1,100 curated tags, which is what lets the ranker reason about content beyond coarse genres."));
children.push(p([B("Why this scale is interesting: "), T("25 million ratings and 30 million genome scores will not fit comfortably in memory on an 8 GB machine if handled naively. A recurring theme of this project — see Section 10 — is doing the heavy reductions (streaming, filtering, sampling) before materializing wide matrices, so that the same pipeline a large team would run on a cluster also runs on a personal laptop.")]));

// ===========================================================================
// 2. Two-stage architecture
// ===========================================================================
children.push(h1("2. Two-Stage Architecture: Retrieve, then Rank"));
children.push(p("The system is built as two stages in sequence: a cheap, recall-oriented retriever followed by an expensive, precision-oriented ranker. This retrieve-then-rank pattern is the industry standard for large-scale recommenders (YouTube, big-tech feed and ads systems all use a candidate-generation stage feeding a heavier ranking stage). Understanding why is the single most important idea in the project."));
children.push(p([B("The economics. "), T("There are ~18,000 eligible items. Scoring every (user, item) pair with a rich 73-feature gradient-boosted model for every request would be wasteful — most items are obviously irrelevant to a given user. So Stage 1 uses a deliberately cheap model to shrink 18,000 items down to 200 strong candidates. Stage 2 then spends its expensive feature computation and model capacity only on those 200, where precision actually matters.")]));
children.push(table([2300, 3530, 3530], [
  ["", "Stage 1 — Retrieval", "Stage 2 — Ranking"],
  ["Model", "ALS (matrix factorization)", "LightGBM LambdaRank"],
  ["Objective", "Maximize recall cheaply", "Maximize precision / NDCG"],
  ["Scope", "All ~18k items", "200 candidates per user"],
  ["Cost", "~2 ms (one matrix-vector product)", "~11 ms (73 features × 200 rows)"],
  ["Failure mode it avoids", "Missing good items entirely", "Showing a bad order"],
]));
children.push(p("The division of labor is the whole point. Stage 1 only has to ensure the good items are somewhere in the 200 — it does not have to order them well. Stage 2 only has to order 200 items it is handed — it never has to consider the other 17,800. Each stage is optimized for the one thing it is good at, and the metric each is tuned against (Recall@200 for Stage 1, NDCG for Stage 2) reflects that. Section 8 shows the striking empirical payoff: the retriever alone actually ranks worse than a trivial popularity list, yet adding the re-ranker on top of it beats every baseline. That gap is the argument for two stages made concrete."));

// ===========================================================================
// 3. Temporal split & leakage
// ===========================================================================
children.push(h1("3. Temporal Split and Avoiding Leakage"));
children.push(p("The data is split by time, not at random:"));
children.push(table([2600, 3380, 3380], [
  ["Split", "Time window", "Role"],
  ["Train", "timestamp < 2017-01-01", "Fit ALS, compute all features"],
  ["Validation", "2017 (whole year)", "Tune retrieval, early-stop the ranker"],
  ["Test", "timestamp ≥ 2018-01-01", "Final, untouched model comparison"],
]));
children.push(p([B("Why random splits leak the future. "), T("If you split ratings randomly, a movie a user rates in 2018 can land in the training set and shape a recommendation you then evaluate against a 2016 interaction. The model gets to peek at the future. That inflates offline metrics and produces a system that looks excellent in a notebook and disappoints in production, because at serving time the future genuinely is unknown. A temporal split mirrors reality: you only ever know the past. Every user and item feature in this project is therefore computed on the train period only — never on validation or test data — so there is no way for future information to bleed into a feature.")]));
children.push(p([B("Warm-start evaluation, and honest cold-start exclusion. "), T("Collaborative-filtering models like ALS cannot say anything meaningful about a user or item they have never seen. So the project keeps only users with ≥ 5 train ratings and items with ≥ 10 train ratings (the "), M("MIN_USER_TRAIN_RATINGS"), T(" / "), M("MIN_ITEM_TRAIN_RATINGS"), T(" thresholds in config). This is a warm-start evaluation: it measures how well the system serves users and items it has evidence about. Cold-start users and items are not silently dropped to flatter the numbers — they are explicitly excluded and documented, with the honest note that a production system would serve them with a popularity plus content-based fallback (Section 11). Reporting on a warm population and being clear about it is more trustworthy than quietly mixing the two.")]));

// ===========================================================================
// 4. Feature engineering
// ===========================================================================
children.push(h1("4. Feature Engineering"));
children.push(p("The ranker sees 73 features per (user, item) candidate, grouped into user features, item features, and cross features. All are computed on the train period only."));
children.push(h2("4.1 Per-user features"));
children.push(bullet([B("n_ratings"), T(" — how active the user is (their total train ratings).")]));
children.push(bullet([B("avg_rating / std_rating"), T(" — the user's mean rating and its spread (a harsh rater vs. a generous one; a consistent vs. an erratic one).")]));
children.push(bullet([B("days_since_last"), T(" — recency: how long before the train cutoff the user was last active. A strong activity/recency signal in the importance table.")]));
children.push(bullet([B("pct_high_ratings"), T(" — the fraction of the user's ratings that are high, a proxy for how easily pleased they are.")]));
children.push(bullet([B("20 genre-affinity fractions"), T(" — for each genre, the fraction of the user's ratings that fell on movies of that genre. This is the user's taste fingerprint and feeds the most important cross-feature.")]));
children.push(h2("4.2 Per-item features"));
children.push(bullet([B("release_year"), T(" — parsed from the title with a regular expression (titles embed the year in parentheses).")]));
children.push(bullet([B("n_ratings / std"), T(" — item popularity and rating spread.")]));
children.push(bullet([B("20 binary genre flags"), T(" — one 0/1 column per genre.")]));
children.push(bullet([B("top-20 genome tags"), T(" — the item's relevance against the 20 globally most informative genome tags (see 4.3).")]));
children.push(p([B("Bayesian average rating (shrinkage). "), T("The headline item feature is a shrunk mean rating, not the raw mean. A movie rated 5.0 by two people is not actually better than a movie rated 4.6 by ten thousand people — the raw mean of the first is high purely because the sample is tiny and noisy. The Bayesian average pulls each item's mean toward the global mean in proportion to how little evidence it has:")]));
children.push(code("bayes_avg = (n * item_mean + C * global_mean) / (n + C)      with C = 25"));
children.push(p([T("Here "), M("n"), T(" is the item's rating count and "), M("C"), T(" (= "), M("BAYES_MIN_COUNT"), T(", 25) is a prior count — you can read it as \"pretend every item starts with 25 ratings at the global average.\" When "), M("n"), T(" is large, the term dominates and the estimate is essentially the item's own mean; when "), M("n"), T(" is small, the global mean dominates and the estimate is conservatively pulled toward average. This shrinkage is what stops obscure, barely-rated items from being recommended just because a handful of fans gave them five stars — exactly the failure a portfolio reviewer would probe.")]));
children.push(h2("4.3 Genome tags under a memory budget"));
children.push(p([T("The genome-scores file is ~30 million rows; pivoting it directly to a wide movie × 1,100-tag matrix would blow the 8 GB budget. The fix is to reduce "), B("before"), T(" pivoting. The file is streamed, the mean relevance of each tag across all movies is accumulated, the top 20 tags by mean relevance are selected, and only those 20 columns are kept and pivoted (see "), M("pipeline/build_genome_pivot.py"), T("). Filtering before the pivot — rather than building the full matrix and then selecting columns — is the difference between fitting in memory and not.")]));
children.push(h2("4.4 Cross features"));
children.push(p("Cross features combine user and item, and are where personalization is actually expressed. There are four:"));
children.push(bullet([B("genre_match"), T(" — the dot product of the user's 20 genre-affinity fractions with the item's 20 genre flags. High when the item's genres line up with what the user habitually watches. The second most important feature overall.")]));
children.push(bullet([B("rating_delta"), T(" — the gap between the item's (Bayesian) average rating and the user's own average rating, normalizing for the fact that some users simply rate everything higher.")]));
children.push(bullet([B("popularity_rank"), T(" — the item's popularity expressed as a rank, giving the model a calibrated notion of \"how mainstream\" a candidate is.")]));
children.push(bullet([B("user_item_era_diff"), T(" — the gap between the item's release era and the era the user tends to watch, capturing whether someone prefers classics or new releases.")]));
children.push(p([T("Counting it up — user features, item features (including the 20 genome columns and 20 genre flags), and the four cross features — gives "), B("73 features total"), T(". Crucially, the same feature builder ("), M("build_matrix"), T(" in "), M("eval/featbuild.py"), T(") is used at train, eval, and serve time, which is what guarantees there is no train/serve skew (Section 9).")]));

// ===========================================================================
// 5. Stage 1 — ALS retrieval
// ===========================================================================
children.push(h1("5. Stage 1 — ALS Retrieval"));
children.push(p([B("Implicit feedback and confidence weighting. "), T("ALS (Alternating Least Squares) here is the implicit-feedback variant. Rather than treating ratings as targets to predict, it treats every interaction as evidence of preference with a confidence that scales with the rating: ")]));
children.push(code("confidence = 1 + alpha * rating       (alpha = 10)"));
children.push(p([T("A 5-star interaction therefore carries far more weight than a 1-star one, while the absence of an interaction is treated as weak, uncertain negative evidence — not a hard \"dislike.\" That is the right model for implicit data: a user not having watched a film usually means they have not encountered it, not that they hate it. The model is "), M("AlternatingLeastSquares(factors=128, iterations=20, regularization=0.01)"), T(" from the "), M("implicit"), T(" library. It factorizes the user × item confidence matrix into a 128-dimensional latent vector per user and per item.")]));
children.push(p([B("Retrieval as an inner-product search. "), T("To get candidates for a user, the system computes the dot product of that user's latent vector against every item vector and takes the top 200: "), M("scores = item_vectors @ user_vec"), T(", then an argpartition for the top-k. Two choices materially improve recall:")]));
children.push(bullet([B("Raw dot product, not cosine. "), T("L2-normalizing the vectors would optimize cosine similarity, but ALS is trained on the raw dot product. Searching with the same inner product the model was trained on keeps retrieval consistent with the training objective. Switching to cosine would silently optimize a different quantity than the one ALS learned.")]));
children.push(bullet([B("Excluding already-seen items. "), T("Each user's train-rated movies are filtered out of the candidate set so all 200 slots go to novel items. The search over-fetches by the number of seen items so that exactly 200 unseen candidates survive. This alone added about 8 points of recall — recommending a movie the user already rated is wasted inventory.")]));
children.push(p([B("The result: Recall@200 = 0.297"), T(" on the validation set (the fraction of a user's future positives captured in the 200 candidates), up from a 0.187 naive baseline. On a strict temporal split where the evaluation items are genuinely future ratings, ~0.30 is the honest, healthy number; the original plan's 0.50 target was unvalidated and is reported as-is rather than reverse-engineered.")]));
children.push(h2("5.1 Why brute-force numpy instead of Faiss"));
children.push(p("An approximate nearest-neighbor index like Faiss exists to make top-k search fast when there are millions of vectors. Here there are only ~18,000 items, so an exact brute-force matrix-vector product takes about 2 ms — Faiss would add complexity and approximation error to buy a speedup that does not exist at this scale, and the exact search is strictly better quality."));
children.push(p([B("There is also a hard blocker. "), T("faiss-cpu and lightgbm each statically link their own copy of the OpenMP runtime (libomp). When both libraries are imported into a single Python process — which is exactly what the evaluation and serving paths do, since they run retrieval and ranking together — the two libomp copies double-initialize and the process deadlocks (a well-known issue on macOS / Apple Silicon). Replacing Faiss with a few lines of numpy sidesteps the conflict entirely, while being faster-to-reason-about, exact, and dependency-light. This is a good example of an engineering decision driven as much by a real runtime constraint as by performance.")]));

// ===========================================================================
// 6. Stage 2 — LightGBM LambdaRank
// ===========================================================================
children.push(h1("6. Stage 2 — LightGBM LambdaRank"));
children.push(p([B("Learning to rank, not to predict. "), T("Stage 2 is a learning-to-rank problem. A pointwise approach (predict each item's rating with MSE, then sort) optimizes a quantity nobody sees — the exact rating — and is indifferent to ordering: it is equally \"happy\" mispredicting an item at position 1 or position 100. LambdaRank instead optimizes NDCG directly. It weights each pairwise swap of two candidates by how much that swap would change NDCG, so the model spends its capacity getting the "), B("top of the list"), T(" right, which is the only part a user ever sees. That is exactly the objective a recommender should care about.")]));
children.push(h2("6.1 Training data and labels"));
children.push(p([T("For each sampled user, the ALS top-200 candidates become one ranking group. The label for a candidate is "), M("min(4, floor(train_rating))"), T(" for items the user rated in train, and 0 otherwise. Two details are deliberate:")]));
children.push(bullet([B("Training candidates are NOT seen-filtered. "), T("Unlike serving, training keeps the user's train-rated items in the candidate set — they are the only source of positive labels. Seen-filtering them away at train time would leave almost no positives to learn from. About 36% of training labels end up positive, a healthy ratio.")]));
children.push(bullet([B("Graded relevance with label_gain [0, 1, 3, 7, 15]. "), T("Labels 0–4 map to exponentially increasing gains, so a 5-star item is worth far more at the top than a 3-star one — matching how NDCG rewards getting the very best items into the very top slots.")]));
children.push(p([T("Key hyperparameters: "), M("objective=lambdarank"), T(", "), M("metric=ndcg"), T(", "), M("num_leaves=63"), T(", "), M("learning_rate=0.05"), T(".")]));
children.push(h2("6.2 The early-stopping fix"));
children.push(p([T("Early stopping needs a validation signal to decide when to stop adding trees. The natural choice — the temporal validation split — turned out to be "), B("too sparse"), T(": most users in that window have very few labeled positives among their 200 candidates, so the NDCG signal was noisy and the model early-stopped at just "), B("2 trees"), T(" — essentially untrained. The fix is to early-stop on a held-out slice of "), B("train-labeled"), T(" users instead, where labels are dense and the NDCG signal is strong. With that change the model trained to roughly "), B("490 trees"), T(" before stopping. The temporal val/test splits are then reserved entirely for the final, unbiased model comparison in Section 7 — they are never used to tune the model, which keeps the comparison honest.")]));
children.push(p([B("Memory and caching. "), T("To fit 8 GB, the ranker trains on a 20,000-user sample (≈4M candidate rows). The built feature matrices are cached to an .npz file, so LightGBM hyperparameters can be re-tuned in seconds without repeating the ~14-minute candidate-building step.")]));

// ===========================================================================
// 7. Offline evaluation
// ===========================================================================
children.push(h1("7. Offline Evaluation"));
children.push(p("Good metrics are what separate a credible recommender from a demo. The project reports proper ranking metrics, plus two beyond-accuracy metrics that catch failure modes raw accuracy hides:"));
children.push(bullet([B("NDCG@k (Normalized Discounted Cumulative Gain). "), T("Sums the relevance of recommended items, discounted by position (an item at rank 1 counts more than one at rank 10), then normalizes against the best possible ordering. The flagship ranking metric: it rewards putting the most relevant items highest.")]));
children.push(bullet([B("Recall@k. "), T("Of all the items the user actually liked, what fraction appear in the top-k. Measures coverage of the user's true positives.")]));
children.push(bullet([B("Precision@k. "), T("Of the k items shown, what fraction the user actually liked. Measures how much of the short list is \"wasted.\"")]));
children.push(bullet([B("MAP@k (Mean Average Precision). "), T("Rewards getting relevant items in early and in good order, averaged across users.")]));
children.push(bullet([B("Coverage@10. "), T("How much of the catalog the system ever recommends across all users. Low coverage means everyone sees the same handful of movies.")]));
children.push(bullet([B("Personalization@10. "), T("How dissimilar different users' lists are from each other. High personalization means the system actually tailors recommendations rather than showing one global list.")]));
children.push(p([T("Evaluation runs on "), B("3,646 test users"), T(" who each have at least 3 future positive items (rating ≥ 4.0) in the test window. Candidates are seen-filtered (train-rated items removed, as at serving time), and the labels are the held-out test positives. This protocol mirrors deployment: predict the future from the past, and score against what the user actually went on to like.")]));

// ===========================================================================
// 8. Results
// ===========================================================================
children.push(h1("8. Results"));
children.push(table([2960, 1280, 1280, 1280, 1280, 1280], [
  ["Model", "NDCG@5", "NDCG@10", "Recall@10", "Precision@10", "MAP@10"],
  ["Popularity", "0.0804", "0.0724", "0.0285", "0.0614", "0.0355"],
  ["ALS (Stage 1)", "0.0656", "0.0662", "0.0291", "0.0621", "0.0291"],
  ["ALS + LambdaRank", "0.1230", "0.1148", "0.0495", "0.0991", "0.0589"],
]));
children.push(p([B("The two-stage system wins on every metric. "), T("NDCG@10 improves +58% over the popularity baseline and +73% over ALS-only; Precision@10 and Recall@10 roughly double. That is the headline result: neither a trivial baseline nor the retriever alone comes close to the full pipeline.")]));
children.push(p([B("The most instructive finding: ALS-only is below popularity. "), T("Look at the middle row — on NDCG@10 the ALS retriever (0.0662) actually scores worse than simply showing everyone the globally most popular movies (0.0724). This is not a bug; it is the entire argument for a two-stage design. ALS is a recall-oriented candidate generator: it casts a wide, diverse net (Coverage@10 of 0.22 and a near-perfect Personalization of 0.996, versus popularity's 0.015 coverage), but its raw top-10 ordering is not precise. The Stage-2 LambdaRank re-ranker is exactly what converts that diverse-but-unordered candidate set into a precise list, and only then does the system beat both baselines. Retrieval and ranking are different jobs; trying to make one model do both is what fails.")]));
children.push(p([B("Feature importance confirms there is no leak. "), T("By LightGBM gain, the top five features are n_ratings (0.354), genre_match (0.111), avg_rating (0.066), days_since_last (0.063), and user_item_era_diff (0.040). Item popularity matters most but is far from monopolizing the model — the learned genre_match cross-feature and per-user recency and affinity signals all contribute meaningfully, which is why the re-ranker personalizes instead of collapsing to a popularity list. Just as importantly, no single feature approaches a gain share of 1.0, which is the signature you want: a feature that dominated everything would usually mean a target leak.")]));

// ===========================================================================
// 9. Serving architecture
// ===========================================================================
children.push(h1("9. Serving Architecture"));
children.push(p([T("The inference service is a FastAPI app ("), M("api/main.py"), T(", "), M("api/recommender.py"), T(") that loads every artifact — the ALS model, item vectors, the LightGBM booster, user and item feature tables, and the per-user seen sets — exactly once at startup. Requests then do no disk I/O: a recommendation is one matrix-vector product (Stage 1) plus 200 feature lookups and a booster score (Stage 2), at roughly 13 ms per request.")]));
children.push(p([B("No train/serve skew. "), T("The same "), M("build_matrix"), T(" feature builder is used in training, evaluation, and serving. This is the single most valuable guarantee in the serving design: a model is only as good as the features it sees, and the classic production bug is computing a feature one way offline and another way online. Sharing one code path makes that bug structurally impossible.")]));
children.push(p("The HTTP surface:"));
children.push(table([3600, 5760], [
  ["Endpoint", "Returns"],
  ["GET /recommend/{id}?k=10&stage=both|retrieval", "Top-K recommendations with ranker score and retrieval_score. stage=retrieval returns raw Stage-1 output; stage=both runs the full pipeline."],
  ["GET /users/{id}", "Profile: n_ratings, avg rating, top genres, genre affinity."],
  ["GET /movies/{id}", "Title, genres, Bayesian average rating, release year."],
  ["GET /metrics", "Offline model comparison and feature importance (for the dashboard charts)."],
  ["GET /health", "Liveness check."],
]));
children.push(p([T("A vanilla-JavaScript three-panel dashboard sits in front of the API: a user profile and genre-affinity panel, a live-recommendations panel with a Stage-1 / both toggle (so you can see what re-ranking changes), and an offline model-comparison and feature-importance chart panel. An "), M("nginx"), T(" same-origin proxy serves the static UI and forwards "), M("/api"), T(" calls to FastAPI, and a Docker Compose file brings up both services (api + ui) with one command. Deliberately no heavyweight front-end framework — the dashboard is an inspection tool for the model, not a product UI.")]));

// ===========================================================================
// 10. 8 GB engineering trade-offs
// ===========================================================================
children.push(h1("10. Engineering Within an 8 GB Budget"));
children.push(p("A theme worth calling out in interviews: every memory compromise here is deliberate and documented, and none of them changes the methodology — they change how the same computation is staged so it fits in 8 GB."));
children.push(bullet([B("Stream genome tags, reduce before pivoting. "), T("The ~30M-row genome file is streamed and reduced to the top-20 tags before the wide pivot, never materializing the full movie × 1,100-tag matrix.")]));
children.push(bullet([B("20k-user ranker sample. "), T("The ranker trains on 20,000 sampled users (≈4M rows) rather than all warm users, which is enough to learn stable feature effects without exhausting memory.")]));
children.push(bullet([B("Capped evaluation sample. "), T("Offline metrics are computed on a bounded user sample (3,646 qualifying test users), enough for stable estimates.")]));
children.push(bullet([B("numpy over Faiss. "), T("Exact brute-force search is equivalent at 18k items and avoids the libomp deadlock — lighter and more correct, not a downgrade (Section 5.1).")]));
children.push(bullet([B("npz feature cache. "), T("Built candidate matrices are cached so the ranker can be re-tuned without rebuilding candidates, turning a 14-minute loop into a few seconds.")]));
children.push(p("The point is not that these are clever tricks; it is that each one is a conscious trade-off with a stated reason, which is what distinguishes engineering from improvisation."));

// ===========================================================================
// 11. What I'd do in production
// ===========================================================================
children.push(h1("11. What I Would Do in Production"));
children.push(p("This is a portfolio-grade system, honest about its boundaries. To take it to production I would add:"));
children.push(bullet([B("Cold-start fallback. "), T("New users and items have no ALS vector. I would serve them with a popularity plus content-based blend (genres and genome tags), then transition to the collaborative model as evidence accumulates.")]));
children.push(bullet([B("Online / incremental retraining. "), T("ALS and the ranker are trained once. A real system retrains on a schedule and incrementally folds in new interactions so recommendations track shifting tastes and new releases.")]));
children.push(bullet([B("A/B testing. "), T("Offline NDCG is a proxy; the ground truth is user behavior. I would gate model changes behind online experiments measuring engagement, not just offline metrics.")]));
children.push(bullet([B("A real ANN index and a feature store at larger scale. "), T("Brute-force search is ideal at 18k items, but at millions of items I would move to an ANN index (Faiss / ScaNN) and a feature store to serve precomputed user and item features with low latency and guaranteed train/serve parity.")]));
children.push(bullet([B("Monitoring. "), T("Track latency, recommendation distribution, coverage, and drift, with alerts — so a silently degrading model is caught before users feel it.")]));

// ===========================================================================
// Closing — how to explain it in an interview
// ===========================================================================
children.push(h1("12. How to Explain This Project in an Interview"));
children.push(p("A concise version: I built a two-stage movie recommender on MovieLens 25M. Stage 1 is an implicit-feedback ALS model that retrieves 200 candidates per user with an exact inner-product search; Stage 2 is a LightGBM LambdaRank re-ranker that orders those 200 on 73 user, item, and cross features. It is served by a FastAPI inference API behind a vanilla-JS inspection dashboard, evaluated on a strict temporal split, and deployed with one Docker Compose command."));
children.push(p("If asked what makes it credible rather than a toy:"));
children.push(bullet("The temporal split prevents future leakage; every feature is computed on the train period only."));
children.push(bullet("Evaluation uses proper ranking metrics (NDCG, MAP, Recall, Precision) plus coverage and personalization, on a warm test population, with cold-start honestly excluded."));
children.push(bullet("The two-stage system beats both a popularity baseline and ALS-only on every accuracy metric — and I can explain why ALS-only loses to popularity, which is the whole case for two stages."));
children.push(bullet("Training, evaluation, and serving share one feature builder, so there is no train/serve skew."));
children.push(bullet("Every memory compromise for the 8 GB budget is deliberate and documented."));
children.push(p("If asked what I would improve next, I would be honest: cold-start fallback, online retraining and A/B testing, and an ANN index plus feature store at larger scale (Section 11)."));

// ===========================================================================
// Document assembly
// ===========================================================================
const doc = new Document({
  creator: "Javier Sánchez Esquivel",
  title: "movie-recsys — Technical Deep-Dive",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      // An explicit Normal style so the headings' basedOn="Normal" resolves;
      // without a defined base style, Word and pandoc fail to recognize the
      // headings as headings (and the TOC field cannot resolve them).
      { id: "Normal", name: "Normal", run: { font: "Arial", size: 22 } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: "2E5496" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "movie-recsys — Technical Deep-Dive", size: 16, color: "888888" })],
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 16, color: "888888" }),
                   new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" })],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, "(" + buf.length + " bytes)");
});
