"""Load a model from the MLflow registry by alias and classify new headlines.

usage: python predict_from_mlflow.py            # uses @champion
       python predict_from_mlflow.py challenger # uses @challenger
"""
import sys

import mlflow
import pandas as pd

from config import DATA_DIR, OUT_DIR, REGISTERED_MODEL, TRACKING_URI, to_class


def main():
    alias = sys.argv[1] if len(sys.argv) > 1 else "champion"
    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{REGISTERED_MODEL}@{alias}"
    model = mlflow.pyfunc.load_model(uri)

    df = pd.read_csv(DATA_DIR / "new_headlines.csv")
    out = model.predict(df["text"].tolist())
    df["predicted_topic"] = [to_class(l) for l in out["label"]]
    df["confidence"] = out["score"].round(4).values

    path = OUT_DIR / f"new_headlines_predictions_{alias}.csv"
    df.to_csv(path, index=False)
    print(f"Loaded {uri}\n")
    with pd.option_context("display.max_colwidth", 70, "display.width", 140):
        print(df[["text", "predicted_topic", "confidence"]].to_string(index=False))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
