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
