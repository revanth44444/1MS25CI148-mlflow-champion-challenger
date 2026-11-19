"""Champion vs Challenger on AG News, tracked in MLflow.

Steps
1. Run each model on the same 400 articles (nested MLflow run per model).
2. Log params, metrics, per-class F1, confusion matrix, predictions and the model itself.
3. Apply champion_rule.json and log the decision on the parent run.
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import pipeline

from config import (
    BATCH_SIZE, CLASSES, DATA_DIR, EXPERIMENT, MAX_LENGTH, MODELS, OUT_DIR,
    ROOT, SEED, STUDENT_USN, TRACKING_URI, to_class,
)

LATENCY_SAMPLES = 100  # articles timed one at a time for avg / p95 latency


def _pin(pkg):
    from importlib.metadata import version
    return f"{pkg}=={version(pkg)}"


# Explicit so MLflow doesn't try to pin packages we never use (e.g. torchvision)
PIP_REQS = [_pin(p) for p in ("mlflow", "transformers", "torch", "sentencepiece", "protobuf")]


def f1_key(c):
    return f"f1_{c.replace('/', '_')}"


def load_pipe(hf_id):
    return pipeline("text-classification", model=hf_id, device=-1,
                    truncation=True, max_length=MAX_LENGTH)


def predict_batched(pipe, texts):
    start = time.perf_counter()
    out = pipe(texts, batch_size=BATCH_SIZE)
    total = time.perf_counter() - start
    labels = [to_class(o["label"]) for o in out]
    scores = [float(o["score"]) for o in out]
    return labels, scores, total


def single_item_latency(pipe, texts):
    """Time one article at a time, like a live API request would."""
    times = []
    for t in texts[:LATENCY_SAMPLES]:
        s = time.perf_counter()
        pipe(t)
        times.append((time.perf_counter() - s) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def save_confusion(y_true, y_pred, short):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix: {short}")
    plt.tight_layout()
    path = OUT_DIR / f"confusion_{short}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def evaluate_model(role, spec, df):
    short, hf_id = spec["short"], spec["hf_id"]
    texts, y_true = df["text"].tolist(), df["label_name"].tolist()

    with mlflow.start_run(run_name=f"{role}_{short}", nested=True) as run:
        pipe = load_pipe(hf_id)
        n_params = sum(p.numel() for p in pipe.model.parameters())
        pipe(texts[:2])  # warm-up so the first timing isn't inflated

        y_pred, conf, total_s = predict_batched(pipe, texts)
        avg_ms, p95_ms = single_item_latency(pipe, texts)

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES,
                                                      average="macro", zero_division=0)
        per_class_f1 = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES,
                                                       average=None, zero_division=0)[2]
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1, "macro_precision": p, "macro_recall": r,
            "avg_latency_ms": avg_ms, "p95_latency_ms": p95_ms,
            "total_inference_s": total_s,
            "mean_confidence": float(np.mean(conf)),
        }
        metrics.update({f1_key(c): float(v) for c, v in zip(CLASSES, per_class_f1)})

        mlflow.set_tags({"role": role, "student_usn": STUDENT_USN, "task": "ag_news_topic"})
        mlflow.log_params({
            "hf_model_id": hf_id, "n_parameters": n_params, "n_eval": len(df),
            "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH, "seed": SEED,
            "latency_samples": LATENCY_SAMPLES, "device": "cpu",
        })
        mlflow.log_metrics(metrics)

        preds = df[["source_index", "text", "label_name"]].copy()
        preds["pred"], preds["confidence"] = y_pred, conf
        preds["correct"] = preds["pred"] == preds["label_name"]
        pred_path = OUT_DIR / f"predictions_{short}.csv"
        preds.to_csv(pred_path, index=False)
        report_path = OUT_DIR / f"report_{short}.txt"
        report_path.write_text(classification_report(y_true, y_pred, labels=CLASSES, digits=4))
        for path in (pred_path, report_path, save_confusion(y_true, y_pred, short)):
            mlflow.log_artifact(str(path))

        info = mlflow.transformers.log_model(
            transformers_model=pipe, name="model", task="text-classification",
            input_example=["Oil prices jump after supply cuts."],
            pip_requirements=PIP_REQS,
        )
        print(f"[{role}] {hf_id}: macro-F1={f1:.4f} acc={metrics['accuracy']:.4f} "
              f"p95={p95_ms:.1f}ms")
        return {"role": role, "short": short, "hf_id": hf_id, "n_params": n_params,
                "run_id": run.info.run_id, "model_uri": info.model_uri,
                "metrics": metrics, "preds": y_pred}


def decide(champ, chall, rule):
    """Apply champion_rule.json. Returns (winner_role, delta, latency ratio, reasons)."""
    pm = rule["primary_metric"]
    delta = chall["metrics"][pm] - champ["metrics"][pm]
    reasons = [f"{pm}: champion {champ['metrics'][pm]:.4f}, challenger "
               f"{chall['metrics'][pm]:.4f}, delta {delta:+.4f} (need >= {rule['min_improvement']})"]
    ok = delta >= rule["min_improvement"]

    ratio = chall["metrics"]["p95_latency_ms"] / champ["metrics"]["p95_latency_ms"]
    lat_ok = ratio <= rule["guardrails"]["max_p95_latency_ratio"]
    reasons.append(f"p95 latency ratio {ratio:.2f}x (limit "
                   f"{rule['guardrails']['max_p95_latency_ratio']}x): {'pass' if lat_ok else 'FAIL'}")

    worst = min(CLASSES, key=lambda c: chall["metrics"][f1_key(c)] - champ["metrics"][f1_key(c)])
    drop = champ["metrics"][f1_key(worst)] - chall["metrics"][f1_key(worst)]
    cls_ok = drop <= rule["guardrails"]["max_per_class_f1_drop"]
    reasons.append(f"worst per-class change: {worst} F1 {-drop:+.4f} (max drop "
                   f"{rule['guardrails']['max_per_class_f1_drop']}): {'pass' if cls_ok else 'FAIL'}")

    winner = "challenger" if (ok and lat_ok and cls_ok) else "champion"
    return winner, delta, ratio, reasons


def comparison_outputs(df, champ, chall, rule):
    rows = []
    for m, direction in rule["metric_directions"].items():
        a, b = champ["metrics"][m], chall["metrics"][m]
        rows.append({"metric": m, "champion": round(a, 4), "challenger": round(b, 4),
                     "delta_b_minus_a": round(b - a, 4), "better_is": direction})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "results_table.csv", index=False)

    a_ok = np.array(champ["preds"]) == df["label_name"].values
    b_ok = np.array(chall["preds"]) == df["label_name"].values
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    # McNemar exact test: of the articles exactly one model got right, is the split 50/50?
    p_value = binomtest(only_b, only_a + only_b, 0.5).pvalue if (only_a + only_b) else 1.0

    dis = df[["source_index", "text", "label_name"]].copy()
    dis["champion_pred"], dis["challenger_pred"] = champ["preds"], chall["preds"]
    dis = dis[dis["champion_pred"] != dis["challenger_pred"]]
    dis.to_csv(OUT_DIR / "disagreements.csv", index=False)

    keys = ["accuracy", "macro_f1"] + [f1_key(c) for c in CLASSES]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, [champ["metrics"][k] for k in keys], 0.4, label=f"Champion ({champ['short']})")
    ax.bar(x + 0.2, [chall["metrics"][k] for k in keys], 0.4, label=f"Challenger ({chall['short']})")
    ax.set_xticks(x, [k.replace("f1_", "F1 ") for k in keys], rotation=20)
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    ax.set_title(f"Quality on {len(df)} AG News test articles")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "compare_quality.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(np.arange(2) - 0.2, [champ["metrics"]["avg_latency_ms"], chall["metrics"]["avg_latency_ms"]], 0.4, label="avg")
    ax.bar(np.arange(2) + 0.2, [champ["metrics"]["p95_latency_ms"], chall["metrics"]["p95_latency_ms"]], 0.4, label="p95")
    ax.set_xticks(range(2), [champ["short"], chall["short"]])
    ax.set_ylabel("ms per article (batch size 1, CPU)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "compare_latency.png", dpi=130)
    plt.close(fig)

    return table, len(dis), only_a, only_b, p_value


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rule = json.loads((ROOT / "champion_rule.json").read_text())
    df = pd.read_csv(DATA_DIR / "eval_articles.csv")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="compare_agnews") as parent:
        mlflow.set_tags({"student_usn": STUDENT_USN, "run_type": "comparison"})
        mlflow.log_artifact(str(ROOT / "champion_rule.json"))
        mlflow.log_artifact(str(DATA_DIR / "eval_articles.csv"))

        champ = evaluate_model("champion", MODELS["champion"], df)
        chall = evaluate_model("challenger", MODELS["challenger"], df)

        winner, delta, ratio, reasons = decide(champ, chall, rule)
        table, n_dis, only_a, only_b, p_value = comparison_outputs(df, champ, chall, rule)

        mlflow.log_metrics({"delta_macro_f1": delta, "p95_latency_ratio": ratio,
                            "n_disagreements": n_dis, "only_champion_correct": only_a,
                            "only_challenger_correct": only_b, "mcnemar_p_value": p_value})
        mlflow.set_tags({"winner": winner,
                         "winner_model": (chall if winner == "challenger" else champ)["hf_id"],
                         "decision_reasons": " | ".join(reasons)})
        for f in ["results_table.csv", "disagreements.csv", "compare_quality.png", "compare_latency.png"]:
            mlflow.log_artifact(str(OUT_DIR / f))

        decision = {
            "parent_run_id": parent.info.run_id, "winner": winner,
            "reasons": reasons, "mcnemar_p_value": p_value,
            "only_champion_correct": only_a, "only_challenger_correct": only_b,
            "n_disagreements": n_dis,
            "models": {m["role"]: {k: m[k] for k in ("short", "hf_id", "n_params", "run_id", "model_uri")}
                       | {"metrics": m["metrics"]} for m in (champ, chall)},
        }
        (OUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2))
        mlflow.log_artifact(str(OUT_DIR / "decision.json"))

    print("\n" + table.to_string(index=False))
    print(f"\nDisagreements: {n_dis}  (only champion right: {only_a}, only challenger right: {only_b}, "
          f"McNemar p = {p_value:.4f})")
    print("\n".join(reasons))
    print(f"\nDECISION: {winner.upper()} wins -> {decision['models'][winner]['hf_id']}")


if __name__ == "__main__":
    main()
