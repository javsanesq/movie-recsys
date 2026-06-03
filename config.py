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
