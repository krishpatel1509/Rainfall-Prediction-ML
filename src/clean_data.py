#!/usr/bin/env python3
"""Clean the supplied Australian weather data without modifying the raw CSV."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

TARGET = "RainTomorrow"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def clean(raw_path: Path) -> tuple[pd.DataFrame, dict]:
    raw_hash = sha256(raw_path)
    df = pd.read_csv(raw_path, low_memory=False)
    if "Date" not in df or TARGET not in df:
        raise ValueError("Required columns Date and RainTomorrow were not found")
    initial_rows, initial_cols = df.shape
    # Normalize text whitespace and blank tokens; preserve genuine missing values.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA})
    date = pd.to_datetime(df["Date"], errors="coerce")
    target_raw = df[TARGET].str.strip().str.title()
    target = target_raw.map({"No": 0, "Yes": 1})
    invalid_date = int(date.isna().sum())
    invalid_target = int(target.isna().sum())
    keep = date.notna() & target.notna()
    cleaned = df.loc[keep].copy()
    cleaned["Date"] = date.loc[keep].dt.strftime("%Y-%m-%d")
    cleaned[TARGET] = target.loc[keep].astype("int8")
    # Coerce numeric-like inputs to numbers; malformed values become missing and are
    # handled by training-only imputers rather than dropping otherwise useful rows.
    numeric_candidates = [c for c in cleaned.columns if c not in ("Date", TARGET)]
    for col in numeric_candidates:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_rows:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.sort_values(["Date", "Location"], kind="stable", na_position="last").reset_index(drop=True)
    # Emit Yes/No labels for analyst-friendly data; the training script encodes them.
    cleaned[TARGET] = cleaned[TARGET].map({0: "No", 1: "Yes"}).astype("category")
    metadata = {
        "source_file": raw_path.name,
        "raw_sha256_before": raw_hash,
        "raw_sha256_after": sha256(raw_path),
        "raw_unchanged": raw_hash == sha256(raw_path),
        "raw_rows": int(initial_rows), "raw_columns": int(initial_cols),
        "rows_invalid_or_missing_date": invalid_date,
        "rows_invalid_or_missing_target": invalid_target,
        "duplicate_rows_removed_after_required_field_filter": duplicate_rows,
        "cleaned_rows": int(len(cleaned)), "cleaned_columns": int(cleaned.shape[1]),
        "cleaning_policy": [
            "Trim leading/trailing whitespace in text fields and normalize blank tokens to missing.",
            "Parse Date; remove rows with an invalid or missing date.",
            "Normalize RainTomorrow Yes/No to a binary interpretation; remove missing or unsupported target labels.",
            "Remove exact duplicate rows after required-field filtering.",
            "Sort by date and location; retain missing feature values for train-only model imputation.",
            "Do not clip or remove statistical outliers; inspect and report them separately in EDA."
        ],
        "cleaned_date_min": str(cleaned["Date"].min()), "cleaned_date_max": str(cleaned["Date"].max()),
        "remaining_missing_cells": int(cleaned.isna().sum().sum()),
        "target_counts": cleaned[TARGET].value_counts().to_dict(),
    }
    return cleaned, metadata

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="raw_data/weatherAUS.csv")
    ap.add_argument("--output-dir", default="cleaned_data")
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output_dir)
    if not inp.exists():
        raise FileNotFoundError(inp)
    out.mkdir(parents=True, exist_ok=True)
    cleaned, metadata = clean(inp)
    output_csv = out / "weatherAUS_clean.csv"
    cleaned.to_csv(output_csv, index=False)
    metadata["output_file"] = output_csv.name
    metadata["output_sha256"] = sha256(output_csv)
    (out / "cleaning_log.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
