"""Build a balanced evaluation set from the AG News *test* split.

100 articles per class, sampled with a fixed seed so the run is reproducible.
"""
import pandas as pd
from datasets import load_dataset

from config import CLASSES, DATA_DIR, SAMPLES_PER_CLASS, SEED


def main():
    ds = load_dataset("fancyzhx/ag_news", split="test").to_pandas()
    ds["label_name"] = ds["label"].map(dict(enumerate(CLASSES)))
    ds["source_index"] = ds.index

    sample = (
        ds.groupby("label")
        .sample(n=SAMPLES_PER_CLASS, random_state=SEED)
        .sample(frac=1, random_state=SEED)  # shuffle classes together
        .reset_index(drop=True)
    )
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "eval_articles.csv"
    sample[["source_index", "text", "label", "label_name"]].to_csv(out, index=False)
    print(f"Saved {len(sample)} articles to {out}")
    print(sample["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
