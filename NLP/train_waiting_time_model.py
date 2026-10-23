"""
Predict-Care AI
Waiting-Time Model Training

Here I am training the regressors that power the waiting-time half
of the pipeline. The dataset is the real DES-style patient-flow
simulation (Data/patient_flow_cleaned.csv), not synthetic data.

Two targets are trained separately because they model different
processes:

    features : Priority, ArrivalTime, NumDoctors, IsPeakArrival
    target A : DocWaitTime            (time until a doctor is seen)
    target B : TotalTimeInHospital    (total length of stay)

Two RandomForestRegressors are trained, evaluated on a held-out
split, and saved together in Models/waiting_time_model.pkl.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "patient_flow_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "Models"
MODEL_PATH = MODEL_DIR / "waiting_time_model.pkl"
METRICS_PATH = MODEL_DIR / "waiting_time_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = ["Priority", "ArrivalTime", "NumDoctors", "IsPeakArrival"]
TARGET_COLUMNS = ["DocWaitTime", "TotalTimeInHospital"]


# ============================================================
# DATA LOADING
# ============================================================

def load_waiting_time_data():
    """Here I am loading and validating the patient-flow dataset."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Waiting-time dataset not found: {DATA_PATH}\n"
            "Place patient_flow_cleaned.csv in the Data/ folder."
        )

    df = pd.read_csv(DATA_PATH)

    required = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    # Here I am coercing types and dropping rows with unusable values
    # rather than silently imputing, so the model never trains on
    # invented features.
    for col in FEATURE_COLUMNS + TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).copy()
    dropped = before - len(df)

    if df.empty:
        raise ValueError("No usable rows remain after type validation.")

    # Here I am sanity-checking ranges that the model relies on.
    if not df["Priority"].isin([1, 2, 3]).all():
        raise ValueError(
            "Priority must be 1, 2, or 3. "
            f"Found: {sorted(df['Priority'].unique().tolist())}"
        )
    if not df["IsPeakArrival"].isin([0, 1]).all():
        raise ValueError(
            "IsPeakArrival must be 0 or 1. "
            f"Found: {sorted(df['IsPeakArrival'].unique().tolist())}"
        )
    if (df["NumDoctors"] < 1).any():
        raise ValueError("NumDoctors must be at least 1.")
    if (df["ArrivalTime"] < 0).any():
        raise ValueError("ArrivalTime must be non-negative.")
    if (df[TARGET_COLUMNS] < 0).any().any():
        raise ValueError("Target columns must be non-negative.")

    print(f"Loaded {len(df)} rows from {DATA_PATH.relative_to(PROJECT_ROOT)}")
    if dropped:
        print(f"Dropped {dropped} rows with missing or non-numeric values.")

    return df


# ============================================================
# TRAINING
# ============================================================

def train_one_target(X_train, X_test, y_train, y_test, target_name):
    """Here I am training and evaluating a single RandomForest."""

    model = RandomForestRegressor(
        n_estimators=120,
        max_leaf_nodes=1024,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n{target_name}")
    print("-" * len(target_name))
    print(f"  MAE : {mae:.2f} minutes")
    print(f"  R^2 : {r2:.3f}")

    return model, {"mae_minutes": round(mae, 2), "r2": round(r2, 3)}


def main():
    print("PREDICT-CARE AI - WAITING-TIME MODEL TRAINING")
    print("=============================================")

    df = load_waiting_time_data()

    X = df[FEATURE_COLUMNS]
    y_doctor = df["DocWaitTime"]
    y_total = df["TotalTimeInHospital"]

    # Here I am using a single split so both targets are evaluated on
    # the same held-out patients. That is important: it means the two
    # MAE numbers describe the same rows and are directly comparable.
    X_train, X_test, idx_train, idx_test = train_test_split(
        X,
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    y_doctor_train = y_doctor.loc[idx_train]
    y_doctor_test = y_doctor.loc[idx_test]
    y_total_train = y_total.loc[idx_train]
    y_total_test = y_total.loc[idx_test]

    print(f"\nTraining rows : {len(idx_train)}")
    print(f"Held-out rows : {len(idx_test)}")

    doctor_model, doctor_metrics = train_one_target(
        X_train, X_test, y_doctor_train, y_doctor_test, "DocWaitTime"
    )
    total_model, total_metrics = train_one_target(
        X_train, X_test, y_total_train, y_total_test, "TotalTimeInHospital"
    )

    # Here I am saving both regressors together so the pipeline only
    # loads one file at startup.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "doctor_model": doctor_model,
            "total_model": total_model,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
        compress=3,
    )

    # Here I am saving the metrics so they can be quoted in the report
    # without re-running training.
    metrics = {
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "rows_total": int(len(df)),
        "rows_train": int(len(idx_train)),
        "rows_test": int(len(idx_test)),
        "random_state": RANDOM_STATE,
        "features": FEATURE_COLUMNS,
        "targets": {
            "DocWaitTime": doctor_metrics,
            "TotalTimeInHospital": total_metrics,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved waiting-time model : {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics            : {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print("\nTraining completed.")


if __name__ == "__main__":
    main()