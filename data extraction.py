import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
DATA_DIR = Path(r"G:\ML\DIgital Health Project\data")
OUTPUT_DIR = Path(r"G:\ML\DIgital Health Project\output")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# HELPER FUNCTIONS
# =========================
def recode_binary(series):
    return series.replace({1: 1, 2: 0, 77: np.nan, 88: np.nan, 99: np.nan})


def safe_extract(df, column):
    """Return column if exists else NaN series"""
    if column in df.columns:
        return df[column]
    else:
        print(f"  ⚠ Missing column skipped: {column}")
        return pd.Series(np.nan, index=df.index)


def safe_max_recode(df, column_list):
    """Use only available columns and recode safely"""
    existing = [c for c in column_list if c in df.columns]

    if len(existing) == 0:
        print(f"  ⚠ None of {column_list} found.")
        return pd.Series(np.nan, index=df.index)

    return recode_binary(df[existing].max(axis=1))


def safe_mean(df, column_list):
    existing = [c for c in column_list if c in df.columns]

    if len(existing) == 0:
        print(f"  ⚠ None of {column_list} found.")
        return pd.Series(np.nan, index=df.index)

    return df[existing].mean(axis=1)


def apply_clinical_exclusions(df):
    log = {}
    n0 = len(df)

    # Confirmed fasting only (b1 must exist)
    if "b1" in df.columns:
        mask = df["b1"] != 2
        log["Not confirmed fasting"] = mask.sum()
        df = df.loc[~mask]
    else:
        log["Not confirmed fasting"] = 0

    # Pregnancy
    if "m5" in df.columns:
        mask = df["m5"] == 1
        log["Pregnant"] = mask.sum()
        df = df.loc[~mask]
    else:
        log["Pregnant"] = 0

    # Target validity
    if "h2" in df.columns:
        mask = df["h2"].isin([77, 88, 99]) | df["h2"].isna()
        log["Invalid or missing target"] = mask.sum()
        df = df.loc[~mask]
    else:
        log["Invalid or missing target"] = 0

    # Physiological plausibility
    phys_mask = (
        (df["Systolic BP"] < 60)
        | (df["Systolic BP"] > 250)
        | (df["BMI"] < 12)
        | (df["BMI"] > 70)
        | (df["Heartrate"] < 30)
        | (df["Heartrate"] > 200)
    )

    log["Physiological outliers"] = phys_mask.sum()
    df = df.loc[~phys_mask]

    log["Total removed"] = n0 - len(df)
    log["Final sample size"] = len(df)

    return df, log


# =========================
# BATCH PROCESSING
# =========================
csv_files = list(DATA_DIR.glob("*.csv"))
assert len(csv_files) > 0, "No CSV files found"

for file in csv_files:
    print(f"\nProcessing: {file.name}")

    df = pd.read_csv(file)
    df.columns = df.columns.str.lower().str.strip()

    out = pd.DataFrame()

    # -------------------------
    # Core variables
    # -------------------------
    out["PID"] = df.iloc[:, 0]
    out["sex"] = safe_extract(df, "sex").replace({"Men": "M", "Women": "F"})
    out["age"] = safe_extract(df, "age")

    # -------------------------
    # Lifestyle
    # -------------------------
    out["Smoking status"] = safe_max_recode(df, ["t1", "t9"])
    out["Alcohol intake"] = safe_max_recode(df, ["a1a", "a1", "a4"])
    out["Physical activity"] = safe_max_recode(df, ["p1", "p4", "p10", "p13"])
    # Sedentary behaviour (continuous variable)
    if "p16a" in df.columns:
        out["Sedentary Behaviour (Avg daily hours)"] = df["p16a"].round(2)
    else:
        print("  ⚠ Sedentary Behaviour skipped (p16a missing)")
        out["Sedentary Behaviour (Avg daily hours)"] = np.nan

    # -------------------------
    # Clinical
    # -------------------------
    out["Diabetes status"] = recode_binary(safe_extract(df, "h7"))

    # BMI (requires m3 & m4)
    if "m3" in df.columns and "m4" in df.columns:
        out["BMI"] = ((df["m4"] / (df["m3"] ** 2)) * 10000).round(2)
    else:
        print("  ⚠ BMI skipped (m3/m4 missing)")
        out["BMI"] = np.nan

    # -------------------------
    # Blood Pressure
    # -------------------------
    out["Systolic BP"] = safe_mean(df, ["m11a", "m12a", "m13a"]).round(2)

    out["Diastolic BP"] = safe_mean(df, ["m11b", "m12b", "m13b"]).round(2)

    # -------------------------
    # Heart Rate
    # -------------------------
    out["Heartrate"] = safe_mean(df, ["m16a", "m16b", "m16c"]).round(2)

    # -------------------------
    # Blood Chemistry
    # -------------------------
    out["Blood Glucose"] = safe_extract(df, "b5")
    out["Total Cholesterol"] = safe_extract(df, "b7")
    out["Triglycerides"] = safe_extract(df, "b8")
    out["HDL Cholesterol"] = safe_extract(df, "b9")

    # -------------------------
    # Target
    # -------------------------
    out["Target (Raised BP)"] = recode_binary(safe_extract(df, "h2"))

    # =========================
    # VALIDATION
    # =========================
    assert set(out["Target (Raised BP)"].dropna().unique()).issubset(
        {0, 1}
    ), "Target encoding error detected"

    # =========================
    # APPLY EXCLUSIONS
    # =========================
    raw_needed = [c for c in ["b1", "m5", "h2"] if c in df.columns]
    combined = pd.concat([df[raw_needed], out], axis=1)

    cleaned, exclusion_log = apply_clinical_exclusions(combined)
    cleaned = cleaned[out.columns]

    # Round all numeric columns to 2 decimal places
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].round(2)

    # SAVE
    # =========================
    output_file = OUTPUT_DIR / f"{file.stem}_cleaned.csv"
    cleaned.to_csv(output_file, index=False)

    print("Exclusion summary:")
    for k, v in exclusion_log.items():
        print(f"  - {k}: {v}")

    print(f"Saved → {output_file.name}")

print("\nBatch preprocessing completed successfully ✅")
