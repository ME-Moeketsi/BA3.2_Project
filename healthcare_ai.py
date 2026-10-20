"""
Predict-Care AI
AI/ML Component

Here I am implementing two separate machine-learning components:
1. KTAS emergency triage classification.
2. Hospital patient-flow waiting-time regression.

The datasets remain separate because they have different targets and represent
different prediction tasks. The KTAS classifier uses only information that can
reasonably be available at the initial triage stage. Post-triage fields are not
used as model inputs because they could cause data leakage.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "Models"

TRIAGE_DATASET = os.path.join(DATA_DIR, "ktas_cleaned.csv")
FLOW_DATASET = os.path.join(DATA_DIR, "patient_flow_cleaned.csv")

TRIAGE_MODEL_FILE = os.path.join(MODEL_DIR, "triage_model.pkl")
# Here I keep separate files for each patient-flow prediction target.
FLOW_MODEL_FILE = os.path.join(MODEL_DIR, "patient_flow_model.pkl")
MEDICATION_MODEL_FILE = os.path.join(MODEL_DIR, "medication_wait_model.pkl")
COMBINED_WAIT_MODEL_FILE = os.path.join(MODEL_DIR, "combined_wait_model.pkl")
TOTAL_TIME_EXCLUDING_MED_WAIT_MODEL_FILE = os.path.join(
    MODEL_DIR, "total_time_excluding_med_wait_model.pkl"
)

# The following file paths are retained for compatibility with the project.
# The triage preprocessor is stored inside the triage model package.
TRIAGE_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "triage_preprocessor.pkl")
FLOW_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "patient_flow_preprocessor.pkl")
MEDICATION_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "medication_wait_preprocessor.pkl")
COMBINED_WAIT_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "combined_wait_preprocessor.pkl")
TOTAL_TIME_EXCLUDING_MED_WAIT_PREPROCESSOR_FILE = os.path.join(
    MODEL_DIR, "total_time_excluding_med_wait_preprocessor.pkl"
)

TRIAGE_TARGET = "KTAS_expert"
# Here I define the three waiting-time prediction targets.
DOCTOR_WAIT_TARGET = "DocWaitTime"
MEDICATION_WAIT_TARGET = "MedWaitTime"
COMBINED_WAIT_TARGET = "CombinedWaitTime"
TOTAL_TIME_EXCLUDING_MED_WAIT_TARGET = "TotalTimeExcludingMedWait"

# These are initial-triage features. Fields recorded after triage, including
# KTAS_RN, Diagnosis in ED, Disposition, Error_group, Length of stay_min,
# KTAS duration_min, and Mistriage, are deliberately excluded.
TRIAGE_CATEGORICAL_FEATURES = [
    "Group",
    "Sex",
    "Arrival mode",
    "Injury",
    "Mental",
    "Pain",
    "Chief_complain",
]

TRIAGE_NUMERICAL_FEATURES = [
    "Age",
    "Patients number per hour",
    "NRS_pain",
    "SBP",
    "DBP",
    "HR",
    "RR",
    "BT",
    "Saturation",
]

TRIAGE_FEATURES = (
    TRIAGE_CATEGORICAL_FEATURES + TRIAGE_NUMERICAL_FEATURES
)
# Here I map the feature names to the columns in my patient-flow dataset.
"""FLOW_FEATURES = [
    "priority_level",
    "IsPeakArrival",
    "num_doctors",
    "PeakLimit",
]"""
#Updated flow features to match the patient flow dataset columns
# Here I select the features available when a patient arrives.
FLOW_NUMERICAL_FEATURES = [
    "Priority",
    "IsPeakArrival",
    "NumDoctors",
    "PeakLimit",
    "ArrivalTime",
]

FLOW_CATEGORICAL_FEATURES = [
    "Scenario",
]

FLOW_FEATURES = FLOW_NUMERICAL_FEATURES + FLOW_CATEGORICAL_FEATURES

# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """Here I make sure the required data and model folders exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Project directories are ready.")


# ============================================================
# DATASET 1 - EMERGENCY TRIAGE
# ============================================================

def load_triage_dataset():
    """Here I load and check the cleaned KTAS dataset."""
    if not os.path.exists(TRIAGE_DATASET):
        raise FileNotFoundError(
            f"Triage dataset not found: {TRIAGE_DATASET}"
        )

    # pandas handles commas inside correctly quoted Chief_complain values.
    data = pd.read_csv(TRIAGE_DATASET)

    if data.empty:
        raise ValueError("The KTAS dataset is empty.")

    if data.columns.duplicated().any():
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicated}")

    print("\n========== TRIAGE DATASET ==========")
    print(f"Records: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Available columns: {len(data.columns)}")

    return data


def preprocess_triage_data(data):
    """
    Here I remove duplicate records and validate the KTAS target.
    Missing feature values are handled by the model's preprocessing pipeline.
    """
    data = data.copy()

    # I remove fully duplicated records without changing the source CSV.
    original_count = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    removed_count = original_count - len(data)

    if TRIAGE_TARGET not in data.columns:
        raise ValueError(
            f"Required target column '{TRIAGE_TARGET}' is missing."
        )

    # I make sure the target is numeric and contains valid KTAS levels 1 to 5.
    data[TRIAGE_TARGET] = pd.to_numeric(
        data[TRIAGE_TARGET], errors="coerce"
    )
    data = data[data[TRIAGE_TARGET].isin([1, 2, 3, 4, 5])].copy()
    data[TRIAGE_TARGET] = data[TRIAGE_TARGET].astype(int)

    if data.empty:
        raise ValueError(
            "No valid records remain after checking KTAS_expert values."
        )

    missing_features = [
        column for column in TRIAGE_FEATURES
        if column not in data.columns
    ]
    if missing_features:
        raise ValueError(
            "The cleaned KTAS dataset is missing required initial-triage "
            f"features: {missing_features}"
        )

    print("\nKTAS dataset validation complete.")
    print(f"Duplicate records removed: {removed_count}")
    print(f"Valid records remaining: {len(data)}")
    print("KTAS target distribution:")
    print(data[TRIAGE_TARGET].value_counts().sort_index().to_string())

    return data


def build_triage_preprocessor():
    """
    Here I create separate preprocessing steps for numerical and categorical
    information. The preprocessing is fitted only on training data.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, TRIAGE_NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, TRIAGE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_triage_data(data, test_size=0.20):
    """
    Here I separate the selected initial-triage features from KTAS_expert.
    The preprocessing is fitted on the training split to avoid data leakage.

    Returns the same six items expected by the existing Main.py workflow:
    X_train, X_test, y_train, y_test, preprocessor, feature_columns.
    """
    data = data.copy()

    missing_columns = [
        column for column in TRIAGE_FEATURES + [TRIAGE_TARGET]
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required KTAS columns: {missing_columns}")

    X = data[TRIAGE_FEATURES].copy()
    y = data[TRIAGE_TARGET].astype(int).copy()

    # I stratify the split so each KTAS class is represented proportionally.
    class_counts = y.value_counts()
    stratify_target = y if len(class_counts) > 1 and class_counts.min() >= 2 else None

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_target,
    )

    preprocessor = build_triage_preprocessor()

    # I learn imputation and encoding rules from training records only.
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    print("\nTriage data prepared.")
    print(f"Training records: {len(X_train_raw)}")
    print(f"Testing records: {len(X_test_raw)}")
    print(f"Input features before encoding: {len(TRIAGE_FEATURES)}")
    print(f"Encoded training feature count: {X_train.shape[1]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        TRIAGE_FEATURES.copy(),
    )


def train_triage_model(X_train, y_train):
    """Here I train a Random Forest classifier for KTAS levels 1 to 5."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("\nTriage classification model trained.")

    return model


def evaluate_triage_model(model, X_test, y_test):
    """Here I evaluate the classifier using held-out triage records."""
    predictions = model.predict(X_test)
    labels = [1, 2, 3, 4, 5]

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== TRIAGE MODEL EVALUATION ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )
    print("\nConfusion Matrix (KTAS 1 through 5):")
    print(confusion_matrix(y_test, predictions, labels=labels))

    return {
        "accuracy": accuracy,
        "predictions": predictions,
    }


# ============================================================
# DATASET 2 - HOSPITAL PATIENT FLOW
# ============================================================

def load_patient_flow_dataset():
    """Here I load the separate hospital patient-flow dataset."""
    if not os.path.exists(FLOW_DATASET):
        raise FileNotFoundError(
            f"Patient-flow dataset not found: {FLOW_DATASET}"
        )

    data = pd.read_csv(FLOW_DATASET)

    if data.empty:
        raise ValueError("The patient-flow dataset is empty.")

    print("\n========== PATIENT FLOW DATASET ==========")
    print(f"Records: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


def preprocess_patient_flow_data(data):
    """Here I remove duplicate rows and define combined waiting time."""
    data = data.copy()
    original_count = len(data)
    data = data.drop_duplicates().reset_index(drop=True)

    required_targets = [
        DOCTOR_WAIT_TARGET,
        MEDICATION_WAIT_TARGET,
        "TotalTimeInHospital",
        "DocServiceTime",
        "MedServiceTime",
    ]
    missing_targets = [column for column in required_targets if column not in data.columns]
    if missing_targets:
        raise ValueError(f"Missing waiting-time target columns: {missing_targets}")

    # Here I convert the timing columns to numeric values before creating targets.
    for column in required_targets:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Here I exclude medication waiting time while retaining both service times.
    # The verified dataset formula is total time = both waits + both service times.
    data[TOTAL_TIME_EXCLUDING_MED_WAIT_TARGET] = (
        data["TotalTimeInHospital"] - data[MEDICATION_WAIT_TARGET]
    )

    # Here I verify that the new target matches the component-based calculation.
    component_total = (
        data[DOCTOR_WAIT_TARGET]
        + data["DocServiceTime"]
        + data["MedServiceTime"]
    )
    valid_comparison = data[TOTAL_TIME_EXCLUDING_MED_WAIT_TARGET].notna() & component_total.notna()
    if not data.loc[valid_comparison, TOTAL_TIME_EXCLUDING_MED_WAIT_TARGET].equals(
        component_total.loc[valid_comparison]
    ):
        if not np.allclose(
            data.loc[valid_comparison, TOTAL_TIME_EXCLUDING_MED_WAIT_TARGET].to_numpy(),
            component_total.loc[valid_comparison].to_numpy(),
        ):
            raise ValueError(
                "TotalTimeExcludingMedWait does not match doctor wait plus both service times."
            )

    # Here I define combined waiting time as doctor wait plus medication wait.
    # This assumes these are separate, non-overlapping waiting periods.
    data[COMBINED_WAIT_TARGET] = data[DOCTOR_WAIT_TARGET] + data[MEDICATION_WAIT_TARGET]

    print(
        "\nPatient-flow records after preprocessing: "
        f"{len(data)} (duplicates removed: {original_count - len(data)})"
    )
    print("Combined waiting time is defined as DocWaitTime + MedWaitTime.")
    print(
        "Total time excluding medication wait is defined as "
        "TotalTimeInHospital - MedWaitTime."
    )
    return data


def prepare_patient_flow_data(data, target_column, test_size=0.20, random_state=42):
    """Here I prepare one waiting-time target and split by simulation run."""
    data = data.copy()
    required_columns = FLOW_FEATURES + [target_column, "RunID"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing patient-flow columns: {missing_columns}")

    for column in FLOW_NUMERICAL_FEATURES + [target_column]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Here I exclude rows missing the target or simulation-run identifier.
    data = data.dropna(subset=[target_column, "RunID"]).copy()
    if data.empty:
        raise ValueError(f"No valid records remain for target '{target_column}'.")

    X = data[FLOW_FEATURES].copy()
    y = data[target_column].copy()
    groups = data["RunID"].copy()

    # Here I keep every simulation run entirely in either training or testing.
    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(group_splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()
    train_runs = set(groups.iloc[train_indices])
    test_runs = set(groups.iloc[test_indices])
    if train_runs.intersection(test_runs):
        raise ValueError("Simulation runs overlap between training and testing.")

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    flow_preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, FLOW_NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, FLOW_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # Here I fit preprocessing only on the training records.
    X_train_processed = flow_preprocessor.fit_transform(X_train)
    X_test_processed = flow_preprocessor.transform(X_test)

    print(f"\nTarget: {target_column}")
    print(f"Training simulation runs: {len(train_runs)}")
    print(f"Testing simulation runs: {len(test_runs)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Encoded feature count: {X_train_processed.shape[1]}")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        flow_preprocessor,
        FLOW_FEATURES.copy(),
    )


def prepare_medication_wait_data(data, test_size=0.20, random_state=42):
    """Here I prepare a group-safe split for the two-stage medication model."""
    required_columns = FLOW_FEATURES + [MEDICATION_WAIT_TARGET, "RunID"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing medication-wait columns: {missing_columns}")

    working_data = data.copy()
    for column in FLOW_NUMERICAL_FEATURES + [MEDICATION_WAIT_TARGET]:
        working_data[column] = pd.to_numeric(working_data[column], errors="coerce")
    working_data = working_data.dropna(subset=[MEDICATION_WAIT_TARGET, "RunID"]).copy()
    if working_data.empty:
        raise ValueError("No valid medication-wait records remain.")
    if (working_data[MEDICATION_WAIT_TARGET] < 0).any():
        raise ValueError("Medication waiting times cannot be negative.")

    X = working_data[FLOW_FEATURES].copy()
    y = working_data[MEDICATION_WAIT_TARGET].astype(float).copy()
    groups = working_data["RunID"].copy()

    # Here I keep each simulation run entirely in either training or testing.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))
    X_train_raw, X_test_raw = X.iloc[train_indices].copy(), X.iloc[test_indices].copy()
    y_train, y_test = y.iloc[train_indices].copy(), y.iloc[test_indices].copy()
    train_runs, test_runs = set(groups.iloc[train_indices]), set(groups.iloc[test_indices])
    if train_runs.intersection(test_runs):
        raise ValueError("Simulation runs overlap between training and testing.")

    # Here I create a binary target to predict whether a medication wait occurs.
    y_train_wait = (y_train > 0).astype(int)
    y_test_wait = (y_test > 0).astype(int)

    def build_preprocessor():
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer(transformers=[
            ("numerical", numeric_pipeline, FLOW_NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, FLOW_CATEGORICAL_FEATURES),
        ], remainder="drop")

    # Here I fit the classifier preprocessor on all training rows only.
    classifier_preprocessor = build_preprocessor()
    X_train_classifier = classifier_preprocessor.fit_transform(X_train_raw)
    X_test_classifier = classifier_preprocessor.transform(X_test_raw)

    # Here I fit the duration regressor only on training rows with positive waits.
    positive_train_mask = y_train > 0
    if not positive_train_mask.any():
        raise ValueError("The training split contains no positive medication waits.")
    X_train_positive_raw = X_train_raw.loc[positive_train_mask].copy()
    y_train_positive = y_train.loc[positive_train_mask].copy()
    duration_preprocessor = build_preprocessor()
    X_train_duration = duration_preprocessor.fit_transform(X_train_positive_raw)

    print("\n========== TWO-STAGE MEDICATION WAIT PREPARATION ==========")
    print(f"Training simulation runs: {len(train_runs)}")
    print(f"Testing simulation runs: {len(test_runs)}")
    print(f"Training records: {len(X_train_raw)}")
    print(f"Testing records: {len(X_test_raw)}")
    print(f"Positive-wait training records: {int(positive_train_mask.sum())}")
    print(f"Positive-wait testing records: {int((y_test > 0).sum())}")
    print(f"Encoded feature count: {X_train_classifier.shape[1]}")

    return {
        "X_train_classifier": X_train_classifier,
        "X_test_classifier": X_test_classifier,
        "y_train_wait": y_train_wait,
        "y_test_wait": y_test_wait,
        "X_train_duration": X_train_duration,
        "y_train_positive": y_train_positive,
        "X_test_raw": X_test_raw,
        "y_test": y_test,
        "classifier_preprocessor": classifier_preprocessor,
        "duration_preprocessor": duration_preprocessor,
        "feature_columns": FLOW_FEATURES.copy(),
    }


def train_two_stage_medication_model(prepared_data):
    """Here I train one classifier and one positive-wait duration regressor."""
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    classifier.fit(prepared_data["X_train_classifier"], prepared_data["y_train_wait"])

    duration_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    duration_model.fit(prepared_data["X_train_duration"], prepared_data["y_train_positive"])
    print("\nTwo-stage medication waiting-time model trained.")
    return classifier, duration_model


def evaluate_two_stage_medication_model(classifier, duration_model, prepared_data):
    """Here I evaluate wait occurrence, total predictions, and positive-wait duration."""
    y_test = prepared_data["y_test"].to_numpy(dtype=float)
    y_test_wait = prepared_data["y_test_wait"].to_numpy(dtype=int)
    X_test_classifier = prepared_data["X_test_classifier"]

    wait_predictions = classifier.predict(X_test_classifier).astype(int)
    duration_predictions = duration_model.predict(
        prepared_data["duration_preprocessor"].transform(prepared_data["X_test_raw"])
    )
    # Here I return zero when no wait is predicted and duration otherwise.
    predictions = np.where(wait_predictions == 1, np.maximum(duration_predictions, 0), 0.0)

    accuracy = accuracy_score(y_test_wait, wait_predictions)
    report = classification_report(y_test_wait, wait_predictions, zero_division=0)
    overall_mae = mean_absolute_error(y_test, predictions)
    overall_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    overall_r2 = r2_score(y_test, predictions) if len(np.unique(y_test)) > 1 else float("nan")
    positive_mask = y_test > 0
    positive_mae = (
        mean_absolute_error(y_test[positive_mask], predictions[positive_mask])
        if positive_mask.any() else float("nan")
    )
    # Here I compare the pipeline with a simple baseline that always predicts zero.
    zero_baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))

    print("\n========== TWO-STAGE MEDICATION WAIT EVALUATION ==========")
    print(f"Wait/no-wait classification accuracy: {accuracy:.4f}")
    print("\nWait/no-wait classification report (0 = no wait, 1 = wait):")
    print(report)
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    print(f"Overall R²: {overall_r2:.4f}")
    print(f"MAE on records with an actual positive wait: {positive_mae:.4f}")
    print(f"Always-zero baseline MAE: {zero_baseline_mae:.4f}")

    return {
        "classification_accuracy": accuracy,
        "mae": overall_mae,
        "rmse": overall_rmse,
        "r2": overall_r2,
        "positive_wait_mae": positive_mae,
        "zero_baseline_mae": zero_baseline_mae,
        "predictions": predictions,
    }


def save_two_stage_medication_model(classifier, duration_model, prepared_data):
    """Here I save both medication models and their matching preprocessors together."""
    package = {
        "model_type": "two_stage_medication_wait",
        "classifier": classifier,
        "duration_model": duration_model,
        "classifier_preprocessor": prepared_data["classifier_preprocessor"],
        "duration_preprocessor": prepared_data["duration_preprocessor"],
        "feature_columns": list(prepared_data["feature_columns"]),
        "target_column": MEDICATION_WAIT_TARGET,
        "zero_means_no_wait": True,
    }
    joblib.dump(package, MEDICATION_MODEL_FILE)
    joblib.dump({
        "classifier_preprocessor": prepared_data["classifier_preprocessor"],
        "duration_preprocessor": prepared_data["duration_preprocessor"],
    }, MEDICATION_PREPROCESSOR_FILE)
    print(f"Two-stage medication model saved to: {MEDICATION_MODEL_FILE}")
    print(f"Medication preprocessors saved to: {MEDICATION_PREPROCESSOR_FILE}")


def predict_medication_wait(input_data, model_package=None):
    """Here I predict medication waiting time for one or more input records."""
    if model_package is None:
        model_package = load_medication_wait_model()
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a dictionary or pandas DataFrame.")

    missing_features = [column for column in model_package["feature_columns"] if column not in input_data.columns]
    if missing_features:
        raise ValueError(f"Missing medication prediction features: {missing_features}")

    X = input_data[model_package["feature_columns"]].copy()
    X_classifier = model_package["classifier_preprocessor"].transform(X)
    wait_flags = model_package["classifier"].predict(X_classifier).astype(int)
    X_duration = model_package["duration_preprocessor"].transform(X)
    durations = model_package["duration_model"].predict(X_duration)
    return np.where(wait_flags == 1, np.maximum(durations, 0.0), 0.0)


def train_patient_flow_model(X_train, y_train):
    """Here I train a Random Forest regressor to estimate waiting time."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("\nPatient-flow regression model trained.")

    return model


def evaluate_patient_flow_model(model, X_test, y_test, target_name="Waiting time"):
    """Here I measure the error and fit for one waiting-time target."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n========== {target_name.upper()} MODEL EVALUATION ==========")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
    }


# ============================================================
# MODEL SAVING
# ============================================================

def save_triage_model(model, scaler, feature_columns):
    """
    Here I save the classifier together with its preprocessing object.
    The 'scaler' key is retained for compatibility with older project code;
    it now contains the fitted triage preprocessor, not a StandardScaler.
    """
    package = {
        "model": model,
        "preprocessor": scaler,
        "scaler": scaler,
        "feature_columns": list(feature_columns),
        "target_column": TRIAGE_TARGET,
        "categorical_features": TRIAGE_CATEGORICAL_FEATURES.copy(),
        "numerical_features": TRIAGE_NUMERICAL_FEATURES.copy(),
        "classes": [1, 2, 3, 4, 5],
    }

    joblib.dump(package, TRIAGE_MODEL_FILE)
    joblib.dump(scaler, TRIAGE_PREPROCESSOR_FILE)

    print(f"\nTriage model saved to: {TRIAGE_MODEL_FILE}")
    print(f"Triage preprocessor saved to: {TRIAGE_PREPROCESSOR_FILE}")


def save_patient_flow_model(model, preprocessor, feature_columns, target_column,
                            model_file=FLOW_MODEL_FILE,
                            preprocessor_file=FLOW_PREPROCESSOR_FILE):
    """Here I save one waiting-time model with its own fitted preprocessor."""
    package = {
        "model": model,
        "preprocessor": preprocessor,
        "scaler": preprocessor,  # Retained for compatibility with older code.
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "numerical_features": FLOW_NUMERICAL_FEATURES.copy(),
        "categorical_features": FLOW_CATEGORICAL_FEATURES.copy(),
    }
    joblib.dump(package, model_file)
    joblib.dump(preprocessor, preprocessor_file)
    print(f"Model saved to: {model_file}")
    print(f"Preprocessor saved to: {preprocessor_file}")


# ============================================================
# MODEL LOADING
# ============================================================

def load_triage_model():
    """Here I load the saved triage model package."""
    if not os.path.exists(TRIAGE_MODEL_FILE):
        raise FileNotFoundError(
            f"Saved triage model was not found: {TRIAGE_MODEL_FILE}"
        )

    return joblib.load(TRIAGE_MODEL_FILE)


def load_patient_flow_model():
    """Here I load the saved patient-flow model package."""
    if not os.path.exists(FLOW_MODEL_FILE):
        raise FileNotFoundError(
            f"Saved patient-flow model was not found: {FLOW_MODEL_FILE}"
        )

    return joblib.load(FLOW_MODEL_FILE)


def load_medication_wait_model():
    """Here I load the saved medication waiting-time model package."""
    if not os.path.exists(MEDICATION_MODEL_FILE):
        raise FileNotFoundError(f"Saved medication model was not found: {MEDICATION_MODEL_FILE}")
    return joblib.load(MEDICATION_MODEL_FILE)


def load_combined_wait_model():
    """Here I load the saved combined waiting-time model package."""
    if not os.path.exists(COMBINED_WAIT_MODEL_FILE):
        raise FileNotFoundError(f"Saved combined waiting model was not found: {COMBINED_WAIT_MODEL_FILE}")
    return joblib.load(COMBINED_WAIT_MODEL_FILE)


def load_total_time_excluding_med_wait_model():
    """Here I load the saved total-time model that excludes medication waiting."""
    if not os.path.exists(TOTAL_TIME_EXCLUDING_MED_WAIT_MODEL_FILE):
        raise FileNotFoundError(
            "Saved total-time-excluding-medication-wait model was not found: "
            f"{TOTAL_TIME_EXCLUDING_MED_WAIT_MODEL_FILE}"
        )
    return joblib.load(TOTAL_TIME_EXCLUDING_MED_WAIT_MODEL_FILE)


# ============================================================
# SYSTEM STATUS
# ============================================================

def system_status():
    """Here I display the current status of the project components."""
    print("\n========== PREDICT-CARE AI STATUS ==========")
    print("Triage Classification       : READY")
    print("Doctor Wait Prediction      : READY")
    print("Medication Wait Prediction  : READY")
    print("Combined Wait Prediction    : READY")
    print("Total Time Excluding Med Wait: READY")
    print("Model Evaluation            : READY")
    print("Prediction                  : READY")
    print("Model Saving                : READY")
    print("Model Loading               : READY")
    print("\nAdditional project components:")
    print("Time-Series Analysis        : SEPARATE MODULE")
    print("Speech Recognition          : SEPARATE MODULE")
    print("Text-to-Speech              : SEPARATE MODULE")
    print("Local LLM / Chatbot         : SEPARATE MODULE")
    print("Deep Learning               : SEPARATE MODULE")
    print("============================================\n")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    """Here I run the complete traditional machine-learning workflow."""
    print("\n============================================")
    print("PREDICT-CARE AI - MACHINE LEARNING WORKFLOW")
    print("============================================")

    create_directories()

    # Here I prepare, train, evaluate, and save the KTAS triage classifier.
    triage_data = load_triage_dataset()
    triage_data = preprocess_triage_data(triage_data)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        triage_preprocessor,
        triage_features,
    ) = prepare_triage_data(triage_data)
    triage_model = train_triage_model(X_train, y_train)
    evaluate_triage_model(triage_model, X_test, y_test)
    save_triage_model(triage_model, triage_preprocessor, triage_features)

    # Here I prepare the patient-flow data once for all prediction targets.
    flow_data = load_patient_flow_dataset()
    flow_data = preprocess_patient_flow_data(flow_data)

    # Here I train the doctor wait, combined wait, and total-time targets separately.
    flow_targets = [
        (DOCTOR_WAIT_TARGET, FLOW_MODEL_FILE, FLOW_PREPROCESSOR_FILE, "Doctor waiting time"),
        (COMBINED_WAIT_TARGET, COMBINED_WAIT_MODEL_FILE, COMBINED_WAIT_PREPROCESSOR_FILE, "Combined waiting time"),
        (
            TOTAL_TIME_EXCLUDING_MED_WAIT_TARGET,
            TOTAL_TIME_EXCLUDING_MED_WAIT_MODEL_FILE,
            TOTAL_TIME_EXCLUDING_MED_WAIT_PREPROCESSOR_FILE,
            "Total hospital time excluding medication waiting",
        ),
    ]

    for target_column, model_file, preprocessor_file, display_name in flow_targets:
        (
            flow_X_train,
            flow_X_test,
            flow_y_train,
            flow_y_test,
            flow_preprocessor,
            flow_features,
        ) = prepare_patient_flow_data(flow_data, target_column)
        flow_model = train_patient_flow_model(flow_X_train, flow_y_train)
        evaluate_patient_flow_model(
            flow_model, flow_X_test, flow_y_test, target_name=display_name
        )
        save_patient_flow_model(
            flow_model,
            flow_preprocessor,
            flow_features,
            target_column,
            model_file=model_file,
            preprocessor_file=preprocessor_file,
        )

    # Here I train the two-stage medication model separately because most waits are zero.
    medication_data = prepare_medication_wait_data(flow_data)
    medication_classifier, medication_duration_model = train_two_stage_medication_model(medication_data)
    evaluate_two_stage_medication_model(
        medication_classifier, medication_duration_model, medication_data
    )
    save_two_stage_medication_model(
        medication_classifier, medication_duration_model, medication_data
    )

    system_status()
    print("Traditional machine-learning workflow completed successfully.")


if __name__ == "__main__":
    main()
