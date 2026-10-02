"""
Predict-Care AI
AI/ML Component

This file contains the machine-learning components of my
Predict-Care AI healthcare decision-support system.

I am using two different datasets:

1. Emergency triage dataset
   - Predicts KTAS expert triage level
   - Random Forest Classifier

2. Hospital patient-flow dataset
   - Predicts doctor waiting time
   - Random Forest Regressor

The two datasets are kept separate because they represent
different healthcare problems and have different targets.

NOTE: The comments will be deleted in the final version of the code. They are here for
debugging purposes.
"""

# Here I am importing the tools I need for my AI system.
import os
import joblib
import numpy as np
import pandas as pd

# Here I am importing tools that help me split and prepare my data.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Here I am importing the measurements I will use to evaluate my models.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Here I am importing the two Random Forest algorithms.
# The classifier predicts categories.
# The regressor predicts numerical values.
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# Here I am defining where my datasets are stored.
DATA_DIR = "data"

# Here I am defining where my trained models will be saved.
MODEL_DIR = "Models"

# Dataset 1: emergency triage data.
TRIAGE_DATASET = os.path.join(
    DATA_DIR,
    "ktas_cleaned.csv"
)

# Dataset 2: hospital patient-flow simulation data.
FLOW_DATASET = os.path.join(
    DATA_DIR,
    "patient_flow_cleaned.csv"
)

# Here I am defining where I will save my trained models.
TRIAGE_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "triage_model.pkl"
)

FLOW_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "patient_flow_model.pkl"
)

# These files store the preprocessing information used by
# each model when making future predictions.
TRIAGE_PREPROCESSOR_FILE = os.path.join(
    MODEL_DIR,
    "triage_preprocessor.pkl"
)

FLOW_PREPROCESSOR_FILE = os.path.join(
    MODEL_DIR,
    "patient_flow_preprocessor.pkl"
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """
    Create the folders required by the project.
    """

    # Here I am making sure my data and model folders exist.
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Project directories are ready.")


# ============================================================
# DATASET 1 - EMERGENCY TRIAGE
# ============================================================

def load_triage_dataset():
    """
    Load the emergency triage dataset.
    """

    if not os.path.exists(TRIAGE_DATASET):
        raise FileNotFoundError(
            f"Triage dataset not found: {TRIAGE_DATASET}"
        )

    # The Kaggle triage dataset uses a semicolon delimiter
    # and cp1254 encoding.
    
    data = pd.read_csv(
        TRIAGE_DATASET
    )

    print("\n========== TRIAGE DATASET ==========")
    print(f"Records: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


def preprocess_triage_data(data):
    """
    Clean the emergency triage dataset.
    """

    # Here I am making a copy so that I do not modify
    # the original dataset.
    data = data.copy()

    # Here I am removing duplicate records.
    data = data.drop_duplicates()

    # Here I am filling missing numerical values
    # using the median of each column.
    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        data[column] = data[column].fillna(
            data[column].median()
        )

    # Here I am filling missing text values.
    text_columns = data.select_dtypes(
        exclude=np.number
    ).columns

    for column in text_columns:
        data[column] = data[column].fillna(
            "Unknown"
        )

    print(
        f"Triage records after preprocessing: {len(data)}"
    )

    return data


def prepare_triage_data(data, test_size=0.20):
    """
    Prepare data for KTAS expert-level classification.

    Target:
        KTAS_expert
    """

    target_column = "KTAS_expert"

    if target_column not in data.columns:
        raise ValueError(
            f"'{target_column}' was not found in the triage dataset."
        )

    # These are the patient features I will initially use.
    # I deliberately avoid post-triage information.
    feature_columns = [
        "Age",
        "Arrival mode",
        "Injury",
        "Mental",
        "Pain",
        "NRS_pain",
        "SBP",
        "DBP",
        "HR",
        "RR",
        "BT",
        "Saturation"
    ]

    # I check that all the features exist before continuing.
    missing_features = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing triage features: {missing_features}"
        )

    # Here I select only the features needed by my model.
    X = data[feature_columns].copy()

    # This is what my model is learning to predict.
    y = data[target_column].copy()

    # Here I convert categorical patient information
    # into numerical columns that the Random Forest can use.
    X = pd.get_dummies(
        X,
        columns=[
            "Arrival mode",
            "Injury",
            "Mental",
            "Pain"
        ],
        dtype=int
    )

    # I save the final feature names so that future
    # predictions use exactly the same structure.
    final_feature_columns = X.columns.tolist()

    # Here I split the data into training and testing data.
    # Stratification helps keep the KTAS class distribution
    # similar between training and testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Here I scale the numerical values.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print("\nTriage data prepared.")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features used: {len(final_feature_columns)}")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        final_feature_columns
    )


def train_triage_model(X_train, y_train):
    """
    Train the Random Forest classification model.
    """

    # Here I create a Random Forest made up of many
    # decision trees to classify the patient's KTAS level.
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # Here the model learns patterns from the training data.
    model.fit(
        X_train,
        y_train
    )

    print("\nTriage classification model trained.")

    return model


def evaluate_triage_model(model, X_test, y_test):
    """
    Evaluate the triage classification model.
    """

    # Here I ask my trained model to predict the
    # KTAS level for patients it has not seen during training.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== TRIAGE MODEL EVALUATION ==========")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return {
        "accuracy": accuracy,
        "predictions": predictions
    }


# ============================================================
# DATASET 2 - HOSPITAL PATIENT FLOW
# ============================================================

def load_patient_flow_dataset():
    """
    Load the hospital patient-flow dataset.
    """

    if not os.path.exists(FLOW_DATASET):
        raise FileNotFoundError(
            f"Patient-flow dataset not found: {FLOW_DATASET}"
        )

    data = pd.read_csv(
        FLOW_DATASET
    )

    print("\n========== PATIENT FLOW DATASET ==========")
    print(f"Records: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


def preprocess_patient_flow_data(data):
    """
    Clean the hospital patient-flow dataset.
    """

    # Here I make a copy so the original dataset remains unchanged.
    data = data.copy()

    # Here I remove duplicate records.
    data = data.drop_duplicates()

    # Here I fill missing numerical values with the median.
    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        data[column] = data[column].fillna(
            data[column].median()
        )

    print(
        f"Patient-flow records after preprocessing: {len(data)}"
    )

    return data


def prepare_patient_flow_data(data, test_size=0.20):
    """
    Prepare the hospital patient-flow data.

    Target:
        DocWaitTime
    """

    target_column = "doc_wait_min"

    if target_column not in data.columns:
        raise ValueError(
            f"'{target_column}' was not found in the patient-flow dataset."
        )

    # These are the operational features I will use.
    # I intentionally do not use IDs or the target itself.
    feature_columns = [
        "priority_level",
        "IsPeakArrival",
        "num_doctors",
        "PeakLimit"
    ]

    missing_features = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing patient-flow features: {missing_features}"
        )

    # Here I select the operational information that
    # can help predict doctor waiting time.
    X = data[feature_columns].copy()

    # This is the numerical value my model will predict.
    y = data[target_column].copy()

    # Here I split the dataset into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    # Here I scale the numerical features.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print("\nPatient-flow data prepared.")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features used: {len(feature_columns)}")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_columns
    )


def train_patient_flow_model(X_train, y_train):
    """
    Train the Random Forest regression model.
    """

    # Here I create a Random Forest regressor because
    # doctor waiting time is a numerical value.
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # Here the model learns the relationship between
    # hospital conditions and doctor waiting time.
    model.fit(
        X_train,
        y_train
    )

    print("\nPatient-flow regression model trained.")

    return model


def evaluate_patient_flow_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate the patient-flow regression model.
    """

    # Here I ask my trained model to predict waiting times
    # for patients it did not see during training.
    predictions = model.predict(X_test)

    # MAE tells me the average size of the prediction error.
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # RMSE gives more weight to larger prediction errors.
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # R-squared tells me how much variation in the target
    # is explained by the model.
    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n========== PATIENT FLOW MODEL EVALUATION ==========")

    print(
        f"MAE: {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²: {r2:.4f}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions
    }


# ============================================================
# MODEL SAVING
# ============================================================

def save_triage_model(
    model,
    scaler,
    feature_columns
):
    """
    Save the triage model and its preprocessing information.
    """

    package = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns
    }

    joblib.dump(
        package,
        TRIAGE_MODEL_FILE
    )

    print(
        f"\nTriage model saved to: {TRIAGE_MODEL_FILE}"
    )


def save_patient_flow_model(
    model,
    scaler,
    feature_columns
):
    """
    Save the patient-flow model and preprocessing information.
    """

    package = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns
    }

    joblib.dump(
        package,
        FLOW_MODEL_FILE
    )

    print(
        f"Patient-flow model saved to: {FLOW_MODEL_FILE}"
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_triage_model():
    """
    Load the saved triage model.
    """

    if not os.path.exists(TRIAGE_MODEL_FILE):
        raise FileNotFoundError(
            "Saved triage model was not found."
        )

    package = joblib.load(
        TRIAGE_MODEL_FILE
    )

    return package


def load_patient_flow_model():
    """
    Load the saved patient-flow model.
    """

    if not os.path.exists(FLOW_MODEL_FILE):
        raise FileNotFoundError(
            "Saved patient-flow model was not found."
        )

    package = joblib.load(
        FLOW_MODEL_FILE
    )

    return package


# ============================================================
# SYSTEM STATUS
# ============================================================

def system_status():
    """
    Display the current status of the Predict-Care AI system.
    """

    print("\n========== PREDICT-CARE AI STATUS ==========")

    print("Triage Classification       : READY")
    print("Patient Flow Regression     : READY")
    print("Model Evaluation            : READY")
    print("Prediction                  : READY")
    print("Model Saving                : READY")
    print("Model Loading               : READY")

    print("\nAdditional project components:")
    print("Time-Series Analysis        : READY")
    print("Speech Recognition          : SEPARATE MODULE")
    print("Text-to-Speech              : SEPARATE MODULE")
    print("Local LLM / Chatbot         : SEPARATE MODULE")
    print("Deep Learning               : READY")

    print("============================================\n")