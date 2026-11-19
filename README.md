# Champion–Challenger: News Topic Classification with MLflow
Revanth Reddy · 1MS25CI148 · Skill Lab (MLOps with MLflow)

## 1. Problem and models
- **Task:** classify a news article into one of 4 topics: **World, Sports, Business, Sci/Tech**.
- **Data:** AG News *test* split (`fancyzhx/ag_news`). 100 articles per class = **400 articles**, sampled with seed 148 by `prepare_data.py` → `data/eval_articles.csv`. Balanced classes so macro and weighted scores mean the same thing.
- **Champion (current production model):** `textattack/albert-base-v2-ag-news`, ALBERT-base, **11.7M parameters**.
- **Challenger:** `textattack/roberta-base-ag-news`, RoBERTa-base, **124.6M parameters** (~11x larger).
- Both models output `LABEL_0..LABEL_3`; `config.to_class()` maps these to the AG News class order.
- Same 400 inputs, same order, CPU only, `max_length=256`, truncation on, batch size 16 for the quality pass.

## 2. Champion rule (committed before the experiment)
The rule lives in `champion_rule.json` and was committed in the first commit, before any model was run. `evaluate.py` reads it and applies it automatically, so the decision cannot be adjusted after seeing the numbers.

1. **Primary metric:** macro-F1 (higher is better). With 4 balanced classes, macro-F1 treats every topic equally.
2. **Minimum improvement:** the challenger must beat the champion by **≥ 0.02 macro-F1** (2 points ≈ 8 articles out of 400). Smaller gains could be noise.
3. **Guardrail 1, latency:** challenger's **p95 latency** must be ≤ **4x** the champion's. p95 is used instead of the mean because slow requests are what users notice.
4. **Guardrail 2, no class left behind:** the challenger must not lose more than **0.03 F1 on any single class**. A model that is better on average but much worse on, say, Business news is not acceptable.
5. **Tie-breaker:** if the rule is not fully met, the champion stays.
6. **Directions:** accuracy / F1 / precision / recall → higher is better; latency and inference time → lower is better.

## 3. How to run
```bash
python3 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

python prepare_data.py            # 1. build the 400-article evaluation set
python evaluate.py                # 2. run both models, log to MLflow, apply the rule
python register_champion.py       # 3. register both models, set @champion / @challenger
python predict_from_mlflow.py     # 4. load models:/news-topic-model@champion and predict
python predict_from_mlflow.py challenger   # (optional) same with the challenger

mlflow ui --backend-store-uri sqlite:///mlflow.db    # open http://127.0.0.1:5000
```
Tracking store: local SQLite (`mlflow.db`), no separate server needed. Experiment: `agnews-topic-champion-challenger`. Registered model: `news-topic-model`.

## 4. What gets logged to MLflow
- **Parent run `compare_agnews`:** champion rule, evaluation data, results table, disagreements, comparison plots, `decision.json`; metrics `delta_macro_f1`, `p95_latency_ratio`, `mcnemar_p_value`; tags `winner`, `winner_model`, `decision_reasons`.
- **Nested runs `champion_albert-agnews` and `challenger_roberta-agnews`:** params (model id, parameter count, batch size, max length, seed…), metrics (accuracy, macro P/R/F1, **per-class F1**, avg and p95 latency, total inference time, mean confidence), artifacts (predictions CSV, classification report, confusion matrix) and the **model itself** (`mlflow.transformers.log_model`).
- **Registry:** each version is tagged with its HF model id, macro-F1 and source run id.

## 5. Results (400 AG News test articles)

| Metric | Champion: ALBERT | Challenger: RoBERTa | Δ (B − A) | Better is |
|---|---|---|---|---|
| Accuracy | 0.9500 | 0.9525 | +0.0025 | higher |
| **Macro-F1 (primary)** | **0.9501** | **0.9526** | **+0.0025** | higher |
| Macro precision | 0.9506 | 0.9533 | +0.0027 | higher |
| Macro recall | 0.9500 | 0.9525 | +0.0025 | higher |
| Avg latency (ms, batch 1) | 50.18 | 39.87 | −10.31 | lower |
| p95 latency (ms, batch 1) | 57.14 | 48.50 | −8.64 (0.85x) | lower |
| Total inference, 400 articles (s) | 19.65 | 11.07 | −8.58 | lower |

**Per-class F1**

| Class | ALBERT | RoBERTa | Δ |
|---|---|---|---|
| World | 0.9500 | 0.9596 | +0.0096 |
| Sports | 0.9899 | 0.9950 | +0.0051 |
| Business | 0.9293 | 0.9239 | **−0.0054** |
| Sci/Tech | 0.9314 | 0.9320 | +0.0006 |

- Correct: ALBERT 380/400, RoBERTa 381/400.
- The models disagree on **13** articles: only ALBERT right on 6, only RoBERTa right on 7 (`outputs/disagreements.csv`).
- **McNemar exact test** on those disagreements: p = 1.00. No evidence that either model is more accurate.
- Business and Sci/Tech are the hardest classes for both models. Half of each model's errors (10 of 20 for ALBERT, 10 of 19 for RoBERTa) are Business ↔ Sci/Tech confusions (see the confusion matrices in `outputs/`).

### Where the models disagree (examples)

| Idx | Article (shortened) | Truth | ALBERT | RoBERTa | Comment |
|---|---|---|---|---|---|
| 1404 | "Panis to Retire from Formula One at End of 2004 … LONDON (Reuters)" | Sports | World | **Sports** | Sports story with a world-news style dateline; ALBERT goes with World. |
| 455 | "Wariner Succeeds Johnson as 400 Meter Champion … ATHENS (Reuters)" | Sports | World | **Sports** | Same pattern: an Olympic story with a foreign dateline. |
| 1062 | "Red Hat replaces CFO …" | Business | **Business** | Sci/Tech | RoBERTa goes with the tech company name; ALBERT gets the executive-change story right. |
| 6240 | "Microsoft and Dell Get Air Force Pact … worth up to $500 million" | Sci/Tech | **Sci/Tech** | Business | Genuinely ambiguous: a tech company signing a money deal. |
| 3627 | "SKorea's Samsung to invest 24 billion dollars in new chip lines (AFP)" | World | Sci/Tech | **World** | The dataset label itself is debatable; it could fit any of three classes. |

Pattern: almost every disagreement is a **tech company + money story** (Business vs Sci/Tech) or a **sports story with a world-news dateline**. Neither model is consistently better; each wins about half of these.

### New, unseen headlines (`data/new_headlines.csv`, written for this project)
10 Indian-context headlines that are not from AG News. Some interesting ones:

| Headline | Champion (ALBERT) | Challenger (RoBERTa) |
|---|---|---|
| RCB clinch thrilling last-over win against Mumbai at Chinnaswamy | Sports ✓ | Business ✗ |
| Researchers find new way to make lithium batteries charge faster | World ✗ | Sci/Tech ✓ |
| Olympic champion announces retirement after knee injury | World ✗ | World ✗ |
| ISRO successfully places weather satellite into orbit from Sriharikota | Sci/Tech ✓ | Sci/Tech ✓ |

Both models stumble on cricket and Indian names (RCB, Chinnaswamy). AG News is ~2004 US/UK news, so this is **domain shift**. Full output: `outputs/new_headlines_predictions_*.csv`.

## 6. Decision
**The Champion stays: `textattack/albert-base-v2-ag-news` remains `@champion`.**

Applying `champion_rule.json` exactly as written:

| Check | Result | Pass? |
|---|---|---|
| Macro-F1 gain ≥ 0.02 | +0.0025 | ✗ **fails** |
| p95 latency ratio ≤ 4x | 0.85x | ✓ |
| No class drops > 0.03 F1 | worst: Business −0.0054 | ✓ |

The challenger passes both guardrails but fails the main condition: it is only 0.25 points better, which is **1 article out of 400**. McNemar's test (7 vs 6 on the disagreements, p = 1.00) confirms this difference is noise. Under the rule, a model is not replaced without a clear gain, so the champion is kept. The decision is logged on the `compare_agnews` run (`winner = champion`, `delta_macro_f1 = 0.0025`), and in the registry **v1 (ALBERT) = `@champion`** and **v2 (RoBERTa) = `@challenger`**.

### Interesting finding: fewer parameters ≠ faster
RoBERTa has **11x more parameters** but was **~15–20% faster** per article on this laptop. ALBERT saves *memory* by sharing the same weights across all 12 layers, but it still runs all 12 layers at full width, so the computation per article is similar to a BERT-base sized model. Parameter count measures model *size*, not *speed*. Because RoBERTa is not slower, the latency guardrail never mattered here.

**What would change the decision?** RoBERTa being faster *and* slightly more accurate makes it a reasonable candidate. A fair next step would be a larger evaluation (e.g. the full 7,600-article test set) to see whether the +0.25 points holds up, plus an evaluation on modern/Indian news where both models struggle.

## 7. Limitations
- **Sample size:** 400 articles. One article = 0.25 points of accuracy.
- **Same-domain test:** both models were fine-tuned on AG News training data and tested on AG News test data. The new-headline test shows they are weaker on other news.
- **Label noise:** some AG News labels are debatable (e.g. the Samsung chip story is labelled World).
- **Latency** was measured once on a single MacBook CPU (100 articles, batch size 1, after warm-up), with the champion run first. Only the ratio between models is meaningful, and it could change on other hardware.
- **No fine-tuning:** both are existing public checkpoints. The project is about comparing, tracking and promoting models, not training them.
- **AI assistance:** I used Claude (Anthropic) as an AI assistant to help write the scripts and this README. I ran the experiments myself and can explain every part of the code.

## Repository structure
```
champion_rule.json       # the promotion rule (committed before the experiment)
config.py                # shared settings: models, paths, MLflow names, label mapping
prepare_data.py          # builds data/eval_articles.csv from AG News test split
evaluate.py              # runs both models, logs everything to MLflow, applies the rule
register_champion.py     # registers both models, sets @champion / @challenger aliases
predict_from_mlflow.py   # loads a model from the registry by alias and predicts
data/       eval_articles.csv, new_headlines.csv
outputs/    predictions_*.csv, report_*.txt, confusion_*.png, compare_*.png,
            results_table.csv, disagreements.csv, decision.json, new_headlines_predictions_*.csv
screenshots/ MLflow UI screenshots (add your own)
```
