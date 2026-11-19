#!/usr/bin/env python3
"""Exploratory analysis for cleaned weatherAUS data; writes figures and summaries."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TARGET = "RainTomorrow"

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="cleaned_data/weatherAUS_clean.csv")
    ap.add_argument("--output-dir", default="outputs/eda")
    args = ap.parse_args()
    path, out = Path(args.input), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path, parse_dates=["Date"])
    if TARGET not in df:
        raise ValueError(f"Missing target {TARGET}")
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    if "RainTomorrow" in numeric: numeric.remove("RainTomorrow")
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = 100 * missing["missing_count"] / len(df)
    missing.sort_values("missing_pct", ascending=False).to_csv(out / "missingness.csv")
    # Target distribution.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["No", "Yes"]
    counts = df[TARGET].value_counts().reindex(order)
    bars = ax.bar(counts.index, counts.values, color=["#3978A8", "#E58C48"], width=.62)
    ax.set(title="Next-day rainfall label distribution", ylabel="Records", xlabel="RainTomorrow")
    ax.bar_label(bars, labels=[f"{v:,} ({v/len(df):.1%})" for v in counts.values], padding=4)
    fig.tight_layout(); fig.savefig(out / "target_distribution.png", dpi=180); plt.close(fig)
    # Missingness in fields with nonzero missing data.
    m = missing[missing.missing_count > 0].sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(m.index, m.missing_pct, color="#3978A8")
    ax.set(title="Missingness by field", xlabel="Missing values (%)", ylabel="")
    ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(out / "missingness.png", dpi=180); plt.close(fig)
    # Feature distribution samples (up to 8 key meteorological variables).
    selected = [c for c in ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed", "Humidity3pm", "Pressure3pm"] if c in numeric]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7)); axes = axes.ravel()
    for ax, col in zip(axes, selected):
        vals = df[col].dropna()
        if col == "Rainfall":
            ax.hist(np.log1p(vals.clip(lower=0)), bins=35, color="#3978A8", alpha=.85)
            ax.set_xlabel("log(1 + rainfall mm)")
        else:
            ax.hist(vals, bins=35, color="#3978A8", alpha=.85)
            ax.set_xlabel(col)
        ax.set_title(col); ax.set_ylabel("Count")
    for ax in axes[len(selected):]: ax.set_visible(False)
    fig.suptitle("Selected numeric feature distributions", y=1.01, fontsize=15)
    fig.tight_layout(); fig.savefig(out / "numeric_distributions.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Pearson correlation heatmap.
    corr_cols = [c for c in numeric if df[c].nunique(dropna=True) > 1]
    corr = df[corr_cols].corr(numeric_only=True)
    corr.to_csv(out / "numeric_correlation.csv")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top_correlations = [{"feature_1":str(a),"feature_2":str(b),"pearson_r":float(v)}
                        for (a,b),v in upper.stack().sort_values(key=lambda s:s.abs(),ascending=False).head(5).items()]
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, linewidths=.15, ax=ax, cbar_kws={"shrink": .75})
    ax.set_title("Pearson correlation among numeric predictors")
    fig.tight_layout(); fig.savefig(out / "correlation_heatmap.png", dpi=180); plt.close(fig)
    # Outliers by Tukey 1.5-IQR fences; counts are descriptive, not filtered.
    outlier_rows=[]
    for col in numeric:
        s=df[col].dropna()
        q1,q3=s.quantile([.25,.75]); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr
        n=int(((s<lo)|(s>hi)).sum()) if iqr>0 else 0
        outlier_rows.append({"feature":col,"q1":q1,"q3":q3,"iqr":iqr,"lower_fence":lo,"upper_fence":hi,"outlier_count":n,"outlier_pct_nonmissing":100*n/len(s) if len(s) else np.nan})
    outliers=pd.DataFrame(outlier_rows).sort_values("outlier_pct_nonmissing",ascending=False)
    outliers.to_csv(out / "outlier_analysis.csv",index=False)
    top=outliers.head(8).sort_values("outlier_pct_nonmissing")
    fig,ax=plt.subplots(figsize=(9,5.2)); ax.barh(top.feature,top.outlier_pct_nonmissing,color="#E58C48")
    ax.set(title="Tukey outlier rate for numeric features",xlabel="Outside 1.5×IQR fences (% of nonmissing)",ylabel="")
    fig.tight_layout(); fig.savefig(out / "outlier_rates.png",dpi=180); plt.close(fig)
    # Temporal seasonality: rainy-label rate by calendar month.
    monthly=df.assign(month=df.Date.dt.month).groupby("month",observed=True)[TARGET].apply(lambda s:(s=="Yes").mean()).reindex(range(1,13))
    monthly.rename("rain_tomorrow_rate").to_csv(out / "monthly_rain_rate.csv")
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(monthly.index,monthly.values,marker="o",lw=2,color="#3978A8")
    ax.set(xticks=range(1,13),xlabel="Calendar month",ylabel="RainTomorrow = Yes rate",title="Observed rain-label rate by month")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "monthly_rain_rate.png",dpi=180); plt.close(fig)
    # Rain today vs tomorrow cross-tab (descriptive association only).
    rain_today_rates={}
    if "RainToday" in df:
        ct=pd.crosstab(df["RainToday"].fillna("Missing"),df[TARGET],normalize="index").reindex(columns=order)
        ct.to_csv(out / "rain_today_vs_tomorrow.csv")
        fig,ax=plt.subplots(figsize=(7,4.5)); ct.plot(kind="bar",stacked=True,ax=ax,color=["#3978A8","#E58C48"])
        ax.set(title="Next-day label by same-day rain label",xlabel="RainToday",ylabel="Within-group proportion")
        ax.legend(title="RainTomorrow",loc="upper right"); fig.tight_layout(); fig.savefig(out / "rain_today_vs_tomorrow.png",dpi=180); plt.close(fig)
        rain_today_rates={str(k):float(v) for k,v in ct["Yes"].drop(index="Missing",errors="ignore").items()}
    counts=df[TARGET].value_counts()
    missing_features=missing.drop(index=["Date",TARGET],errors="ignore").sort_values("missing_pct",ascending=False)
    top_missing=missing_features.head(5)
    top_out=outliers.head(5)
    summary={
        "rows":int(len(df)),"columns":int(df.shape[1]),"date_min":str(df.Date.min().date()),"date_max":str(df.Date.max().date()),
        "target_counts":{str(k):int(v) for k,v in counts.items()},"rain_tomorrow_rate":float((df[TARGET]=="Yes").mean()),
        "numeric_predictors":numeric,"categorical_predictors":[c for c in df.columns if c not in numeric+["Date",TARGET]],
        "top_missing_fields":[{"field":str(i),"missing_pct":float(r.missing_pct)} for i,r in top_missing.iterrows()],
        "highest_tukey_outlier_rates":[{"feature":str(r.feature),"outlier_pct":float(r.outlier_pct_nonmissing)} for r in top_out.itertuples()],
        "top_absolute_pearson_correlations":top_correlations,
        "monthly_rain_rate_min":float(monthly.min()),"monthly_rain_rate_min_month":int(monthly.idxmin()),
        "monthly_rain_rate_max":float(monthly.max()),"monthly_rain_rate_max_month":int(monthly.idxmax()),
        "rain_tomorrow_rate_by_rain_today":rain_today_rates,
        "all_numeric_missing_pct":{str(c):float(missing.loc[c,"missing_pct"]) for c in numeric},
    }
    (out/"eda_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    lines=["# Exploratory Data Analysis — cleaned weatherAUS", "", f"- Rows and columns: **{len(df):,} × {df.shape[1]}**", f"- Observation period: **{summary['date_min']} to {summary['date_max']}**", f"- RainTomorrow=Yes: **{counts.get('Yes',0):,} ({summary['rain_tomorrow_rate']:.2%})**", f"- Predictors: **{len(numeric)} numeric**, **{len(summary['categorical_predictors'])} categorical** (Date excluded from model predictors).", "", "## Findings", "", f"- The label is imbalanced: the positive class is {summary['rain_tomorrow_rate']:.2%}; accuracy should be interpreted together with recall, F1, ROC-AUC, and average precision.", "- Missing feature values are intentionally retained for train-only imputation. Highest missingness: " + ", ".join(f"{r['field']} ({r['missing_pct']:.1f}%)" for r in summary['top_missing_fields']) + ".", "- Tukey 1.5×IQR outlier flags are descriptive; no observations were removed or clipped based on the fences. Highest rates: " + ", ".join(f"{r['feature']} ({r['outlier_pct']:.1f}%)" for r in summary['highest_tukey_outlier_rates']) + ".", f"- Rainfall is strongly right-skewed and has the largest Tukey outlier rate ({summary['highest_tukey_outlier_rates'][0]['outlier_pct']:.1f}% of its nonmissing values); these flagged values are retained as potentially genuine extremes.", "- The strongest pairwise numeric association is " + "; ".join(f"{r['feature_1']}–{r['feature_2']} (r={r['pearson_r']:.3f})" for r in top_correlations[:3]) + ". Correlation is descriptive, not causal.", f"- Monthly RainTomorrow rates range from {summary['monthly_rain_rate_min']:.1%} (month {summary['monthly_rain_rate_min_month']}) to {summary['monthly_rain_rate_max']:.1%} (month {summary['monthly_rain_rate_max_month']}); this pooled seasonal pattern may reflect location and year composition.", "- Among rows with `RainToday=No`, the next-day positive rate is " + f"{rain_today_rates.get('No',float('nan')):.1%}; after `RainToday=Yes`, it is {rain_today_rates.get('Yes',float('nan')):.1%}. This is an association, not a causal estimate.", "", "## Figures and tables", "", "See the adjacent PNG and CSV files: target distribution, missingness, feature distributions, numeric correlation heatmap, Tukey outlier rates, monthly positive-label rate, and RainToday/RainTomorrow cross-tab.", "", "## Interpretation cautions", "", "Outlier fences are univariate and may flag valid meteorological extremes. Pairwise Pearson correlations capture only linear associations and may not reveal nonlinear or conditional relationships."]
    (out/"EDA_findings.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
