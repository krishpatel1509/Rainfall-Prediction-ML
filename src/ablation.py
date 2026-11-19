#!/usr/bin/env python3
"""Ablation study for the selected CatBoost configuration (chronological holdout).

Each variant removes ONE component from the full pipeline and is refit on the whole
training era, then scored once on the later-period holdout. Hyperparameters are fixed
to the configuration chosen by train.py, so differences come from the removed component.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

SEED = 42
TARGET = "RainTomorrow"
BEST = dict(iterations=220, depth=5, learning_rate=0.07, l2_leaf_reg=7)

def build(numeric, categorical, *, indicators=True, class_weight=True):
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=indicators))])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    parts = []
    if numeric: parts.append(("num", num, numeric))
    if categorical: parts.append(("cat", cat, categorical))
    prep = ColumnTransformer(parts, remainder="drop")
    model = CatBoostClassifier(verbose=False, loss_function="Logloss", eval_metric="AUC",
                               random_seed=SEED, allow_writing_files=False, thread_count=4,
                               auto_class_weights="Balanced" if class_weight else None, **BEST)
    return Pipeline([("prep", prep), ("model", model)])

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="cleaned_data/weatherAUS_clean.csv")
    ap.add_argument("--output-dir", default="outputs/ablation")
    ap.add_argument("--test-fraction", type=float, default=0.20)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["Date"])
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Date", TARGET]).sort_values(["Date", "Location"], kind="stable").reset_index(drop=True)
    feats = [c for c in df.columns if c not in ["Date", TARGET]]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feats if c not in numeric]
    dates = np.array(sorted(df.Date.unique()))
    cut = min(max(int(np.floor((1 - args.test_fraction) * len(dates))), 1), len(dates) - 1)
    cutoff = pd.Timestamp(dates[cut - 1])
    train, test = df[df.Date <= cutoff], df[df.Date > cutoff]
    ytr, yte = train[TARGET].astype(int), test[TARGET].astype(int)

    sparse_cols = ["Sunshine", "Evaporation", "Cloud9am", "Cloud3pm"]
    variants = [
        ("Full pipeline (reference)", dict(), numeric, categorical),
        ("w/o class weighting", dict(class_weight=False), numeric, categorical),
        ("w/o missingness indicators", dict(indicators=False), numeric, categorical),
        ("w/o RainToday", dict(), numeric, [c for c in categorical if c != "RainToday"]),
        ("w/o Location", dict(), numeric, [c for c in categorical if c != "Location"]),
        ("w/o wind-direction features", dict(), numeric,
         [c for c in categorical if c not in ("WindGustDir", "WindDir9am", "WindDir3pm")]),
        ("w/o high-missingness numerics*", dict(), [c for c in numeric if c not in sparse_cols], categorical),
        ("w/o 3 pm humidity", dict(), [c for c in numeric if c != "Humidity3pm"], categorical),
    ]
    rows = []
    for name, kw, num, cat in variants:
        pipe = build(num, cat, **kw).fit(train[num + cat], ytr)
        s = pipe.predict_proba(test[num + cat])[:, 1]
        p = (s >= 0.5).astype(int)
        r = dict(variant=name, n_features=len(num) + len(cat),
                 accuracy=accuracy_score(yte, p), precision=precision_score(yte, p, zero_division=0),
                 recall=recall_score(yte, p), f1=f1_score(yte, p),
                 roc_auc=roc_auc_score(yte, s), average_precision=average_precision_score(yte, s))
        rows.append(r)
        print(f"{name:34s} acc={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} AUC={r['roc_auc']:.4f} AP={r['average_precision']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    ref = res.iloc[0]
    res["delta_AP_vs_full"] = res.average_precision - ref.average_precision
    res["delta_F1_vs_full"] = res.f1 - ref.f1
    res.to_csv(out / "ablation_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    r = res.iloc[1:].sort_values("delta_AP_vs_full")
    ax.barh(r.variant, r.delta_AP_vs_full, color=["#C0392B" if v < 0 else "#2E8B57" for v in r.delta_AP_vs_full])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(r.delta_AP_vs_full.min() * 1.30, r.delta_AP_vs_full.max() + 0.008)
    ax.set(xlabel="Change in holdout average precision vs. full pipeline",
           title="Ablation of the selected CatBoost pipeline")
    for y, v in enumerate(r.delta_AP_vs_full):
        if abs(v) > 0.03:   # long bar: label inside so it cannot collide with tick labels
            ax.text(v / 2, y, f"{v:+.4f}", va="center", ha="center", fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(v + (0.0004 if v >= 0 else -0.0004), y, f"{v:+.4f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "ablation_ap_delta.png", dpi=200, bbox_inches="tight")
    (out / "ablation_meta.json").write_text(json.dumps(dict(
        cutoff=str(cutoff.date()), train_rows=len(train), test_rows=len(test), seed=SEED,
        catboost_params=BEST, note="*Sunshine, Evaporation, Cloud9am, Cloud3pm"), indent=2) + "\n")

if __name__ == "__main__":
    main()
