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


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "Models"

TRIAGE_DATASET = os.path.join(DATA_DIR, "ktas_cleaned.csv")
FLOW_DATASET = os.path.join(DATA_DIR, "patient_flow_cleaned.csv")

TRIAGE_MODEL_FILE = os.path.join(MODEL_DIR, "triage_model.pkl")
FLOW_MODEL_FILE = os.path.join(MODEL_DIR, "patient_flow_model.pkl")

# The following file paths are retained for compatibility with the project.
# The triage preprocessor is stored inside the triage model package.
TRIAGE_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "triage_preprocessor.pkl")
FLOW_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "patient_flow_preprocessor.pkl")

TRIAGE_TARGET = "KTAS_expert"
# Here I identify the doctor waiting time that my model must predict.
#FLOW_TARGET = "doc_wait_min"

FLOW_TARGET = "DocWaitTime"

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
FLOW_FEATURES = [
    "Priority",
    "IsPeakArrival",
    "NumDoctors",
    "PeakLimit",
]

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
    """Here I remove duplicate rows and leave missing-value handling to preparation."""
    data = data.copy()
    original_count = len(data)
    data = data.drop_duplicates().reset_index(drop=True)

    print(
        "\nPatient-flow records after preprocessing: "
        f"{len(data)} (duplicates removed: {original_count - len(data)})"
    )

    return data


def prepare_patient_flow_data(data, test_size=0.20):
    """Here I prepare the operational features for waiting-time regression."""
    data = data.copy()

    required_columns = FLOW_FEATURES + [FLOW_TARGET]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing patient-flow columns: {missing_columns}"
        )

    # I convert the selected columns to numeric values and reject invalid targets.
    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=[FLOW_TARGET]).copy()
    if data.empty:
        raise ValueError("No valid patient-flow target values remain.")

    X = data[FLOW_FEATURES].copy()
    y = data[FLOW_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    # I fit imputation and scaling on the training split only. I save both
    # steps together so future predictions use the same preprocessing.
    flow_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_scaled = flow_preprocessor.fit_transform(X_train)
    X_test_scaled = flow_preprocessor.transform(X_test)

    print("\nPatient-flow data prepared.")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features used: {len(FLOW_FEATURES)}")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        flow_preprocessor,
        FLOW_FEATURES.copy(),
    )


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


def evaluate_patient_flow_model(model, X_test, y_test):
    """Here I measure waiting-time prediction error and model fit."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n========== PATIENT FLOW MODEL EVALUATION ==========")
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


def save_patient_flow_model(model, scaler, feature_columns):
    """Here I save the waiting-time model and its fitted preprocessor."""
    package = {
        "model": model,
        "preprocessor": scaler,
        "scaler": scaler,  # Retained for compatibility with existing code.
        "feature_columns": list(feature_columns),
        "target_column": FLOW_TARGET,
    }

    joblib.dump(package, FLOW_MODEL_FILE)
    joblib.dump(scaler, FLOW_PREPROCESSOR_FILE)

    print(f"Patient-flow model saved to: {FLOW_MODEL_FILE}")
    print(f"Patient-flow preprocessor saved to: {FLOW_PREPROCESSOR_FILE}")


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


# ============================================================
# SYSTEM STATUS
# ============================================================

def system_status():
    """Here I display the current status of the project components."""
    print("\n========== PREDICT-CARE AI STATUS ==========")
    print("Triage Classification       : READY")
    print("Patient Flow Regression     : READY")
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

    # Here I separately prepare, train, evaluate, and save the waiting-time model.
    flow_data = load_patient_flow_dataset()
    flow_data = preprocess_patient_flow_data(flow_data)
    (
        flow_X_train,
        flow_X_test,
        flow_y_train,
        flow_y_test,
        flow_preprocessor,
        flow_features,
    ) = prepare_patient_flow_data(flow_data)
    flow_model = train_patient_flow_model(flow_X_train, flow_y_train)
    evaluate_patient_flow_model(flow_model, flow_X_test, flow_y_test)
    save_patient_flow_model(flow_model, flow_preprocessor, flow_features)

    system_status()
    print("Traditional machine-learning workflow completed successfully.")


if __name__ == "__main__":
    main()
