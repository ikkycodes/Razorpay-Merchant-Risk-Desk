"""Part A — Data Foundation: clean + features + audits.
Usage: python prepare.py (run from repo root, needs transactions_raw.csv)
"""
import pandas as pd
import numpy as np


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned copy without modifying input. Order matters."""
    df = raw_df.copy(deep=True)
    # 1. Drop exact duplicate rows
    df = df.drop_duplicates()
    # 2. Strip + title case for merchant_category and city
    for col in ["merchant_category", "city"]:
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].isna(), df[col].astype(str).str.strip().str.title())
    # 3. Drop rows where amount_inr is missing
    df = df.dropna(subset=["amount_inr"])
    # 4. Drop rows where amount_inr < 1 or > 1_000_000
    df = df[(df["amount_inr"] >= 1) & (df["amount_inr"] <= 1_000_000)]
    # 5. Reset index
    df = df.reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy with amount_vs_avg_ratio, is_night, high_value."""
    out = df.copy(deep=True)
    out["amount_vs_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"]
    out["is_night"] = (out["hour_of_day"] <= 5).astype(int)
    out["high_value"] = (out["amount_inr"] > 50000).astype(int)
    return out


def main():
    raw = pd.read_csv("transactions_raw.csv")
    print(f"Raw rows: {len(raw)}, columns: {len(raw.columns)}")
    print(f"Columns: {list(raw.columns)}")
    n_dup = int(raw.duplicated().sum())
    print(f"Exact duplicate rows: {n_dup}")
    dedup = raw.drop_duplicates()
    print(f"Rows after de-duplication: {len(dedup)}")
    print(f"Missing amount_inr after de-dup: {int(dedup['amount_inr'].isna().sum())}")
    print(f"Missing distance_from_home_km after de-dup: {int(dedup['distance_from_home_km'].isna().sum())}")
    # out-of-range among non-missing amounts after dedup
    oor = dedup.dropna(subset=["amount_inr"])
    oor = oor[(oor["amount_inr"] < 1) | (oor["amount_inr"] > 1_000_000)]
    print(f"Out-of-range amounts (<1 or >1000000): {len(oor)}")
    print(f"Distinct raw merchant_category spellings: {dedup['merchant_category'].nunique(dropna=False)}")
    print(f"Distinct raw city spellings: {dedup['city'].nunique(dropna=False)}")
    print(f"Raw merchant_category values: {sorted(dedup['merchant_category'].dropna().unique().tolist())}")
    print(f"Raw city values: {sorted(dedup['city'].dropna().unique().tolist())}")

    cleaned = clean_transactions(raw)
    print(f"\nCleaned rows: {len(cleaned)}")
    print(f"Duplicate rows after clean: {int(cleaned.duplicated().sum())}")
    print("Merchant category counts:")
    print(cleaned["merchant_category"].value_counts().to_string())
    print("City counts:")
    print(cleaned["city"].value_counts().to_string())
    print(f"Distinct cities: {cleaned['city'].nunique()}")
    print("Payment mode counts:")
    print(cleaned["payment_mode"].value_counts().to_string())
    print(f"Min amount: {cleaned['amount_inr'].min():.2f}")
    print(f"Max amount: {cleaned['amount_inr'].max():.2f}")
    print(f"Remaining missing distances: {int(cleaned['distance_from_home_km'].isna().sum())}")

    feat = add_features(cleaned)
    print(f"\nis_night sum: {int(feat['is_night'].sum())}")
    print(f"high_value sum: {int(feat['high_value'].sum())}")
    print(f"max amount_vs_avg_ratio: {feat['amount_vs_avg_ratio'].max():.2f}")

    # Leakage audit
    print("\nLeakage audit (chargeback_filed vs is_fraud):")
    ct = pd.crosstab(cleaned["chargeback_filed"], cleaned["is_fraud"])
    print(ct.to_string())
    for v in [0, 1]:
        sub = cleaned[cleaned["chargeback_filed"] == v]
        print(f"chargeback_filed={v}: rows={len(sub)}, fraud rows={int((sub['is_fraud'] == 1).sum())}")
    n_fraud = int((cleaned["is_fraud"] == 1).sum())
    rate = n_fraud / len(cleaned) * 100
    print(f"Fraud rows: {n_fraud} / {len(cleaned)} = {rate:.2f}%")
    acc = (len(cleaned) - n_fraud) / len(cleaned)
    print(f"All-non-fraud baseline: accuracy={acc:.4f} ({acc*100:.2f}%), recall=0.0000")
    print("Why accuracy is misleading: with 6.18% fraud, predicting all-legit gives ~93.8% accuracy while catching zero frauds; use precision/recall/F1.")
    print("Exclude chargeback_filed (post-outcome leakage), txn_id/customer_id (identifiers, no generalisation); also exclude target is_fraud from inputs.")

    # one-row sanity
    try:
        one = add_features(clean_transactions(raw.head(1).copy()))
        print(f"One-row check OK: shape={one.shape}")
    except Exception as e:
        print(f"One-row check FAILED: {e}")


if __name__ == "__main__":
    main()
