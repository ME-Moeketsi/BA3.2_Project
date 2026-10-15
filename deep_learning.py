"""
Predict-Care AI
Deep Learning Component

Here I am implementing the Deep Learning component of my
Predict-Care AI healthcare decision-support system.

This component predicts the expert KTAS triage level from
patient information available during the initial assessment.

My traditional Machine Learning model remains separate in
healthcare_ai.py. This file uses a neural network so I can
evaluate a second modelling approach.

The model is a research prototype for academic evaluation.
It is not validated for clinical use.
"""

import os

# Here I am selecting PyTorch as the backend before importing Keras.
os.environ["KERAS_BACKEND"] = "torch"

import joblib
import numpy as np
import pandas as pd
import keras

from keras import layers
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PROJECT PATHS AND FEATURES
# ============================================================

DATA_DIR = "Data"
MODEL_DIR = "Models"

TRIAGE_DATASET = os.path.join(DATA_DIR, "ktas_cleaned.csv")

DEEP_LEARNING_MODEL_FILE = os.path.join(
    MODEL_DIR, "deep_learning_triage_model.keras"
)
DEEP_LEARNING_PREPROCESSOR_FILE = os.path.join(
    MODEL_DIR, "deep_learning_triage_preprocessor.pkl"
)

# Here I am using the same initial-assessment features as the
# updated traditional ML component so the two approaches are
# based on comparable input information.
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
TRIAGE_TARGET = "KTAS_expert"
KTAS_CLASSES = [1, 2, 3, 4, 5]
RANDOM_STATE = 42


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

def create_deep_learning_directories():
    """Here I am creating the directory used to save model files."""
    os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_triage_dataset():
    """Here I am loading the cleaned KTAS dataset."""
    if not os.path.exists(TRIAGE_DATASET):
        raise FileNotFoundError(
            f"Dataset not found: {TRIAGE_DATASET}. "
            "Check that ktas_cleaned.csv is inside the Data folder."
        )

    data = pd.read_csv(TRIAGE_DATASET)

    print("\n============================================")
    print("DEEP LEARNING DATASET")
    print("============================================")
    print(f"Dataset path: {TRIAGE_DATASET}")
    print(f"Dataset shape: {data.shape}")

    return data


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_triage_data(data):
    """
    Here I am checking the required columns, cleaning invalid
    numeric values, and keeping only valid KTAS target classes.

    I deliberately exclude KTAS_RN, Diagnosis in ED, Disposition,
    Error_group, Length of stay_min, KTAS duration_min and
    Mistriage from the model inputs because they can reveal
    triage decisions or information recorded after triage.
    """
    required_columns = TRIAGE_FEATURES + [TRIAGE_TARGET]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The KTAS dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    cleaned = data[required_columns].copy()

    # Here I am converting numeric fields and the target to numbers.
    for column in TRIAGE_NUMERICAL_FEATURES + [TRIAGE_TARGET]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Here I am replacing infinite values with missing values.
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    # Here I am removing rows without a valid target label.
    cleaned = cleaned.dropna(subset=[TRIAGE_TARGET])
    cleaned = cleaned[cleaned[TRIAGE_TARGET].isin(KTAS_CLASSES)].copy()
    cleaned[TRIAGE_TARGET] = cleaned[TRIAGE_TARGET].astype(int)

    # Here I am treating blank categorical values as missing.
    for column in TRIAGE_CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].replace(r"^\s*$", np.nan, regex=True)
        cleaned[column] = cleaned[column].astype("object")

    print(f"Records available after target cleaning: {len(cleaned)}")
    print("KTAS target distribution:")
    print(cleaned[TRIAGE_TARGET].value_counts().sort_index().to_string())

    if cleaned.empty:
        raise ValueError("No valid records remain after preprocessing.")

    return cleaned


# ============================================================
# BUILD PREPROCESSING PIPELINE
# ============================================================

def build_deep_learning_preprocessor():
    """
    Here I am building one reusable preprocessing pipeline.

    Numeric values are imputed and standardised. Categorical
    values are imputed and one-hot encoded. I fit this pipeline
    on training data only to avoid data leakage.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                TRIAGE_NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                TRIAGE_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ============================================================
# PREPARE DATA FOR THE NEURAL NETWORK
# ============================================================

def prepare_deep_learning_data(
    data,
    test_size=0.20,
    random_state=RANDOM_STATE,
):
    """
    Here I am splitting the data before fitting preprocessing.

    KTAS classes 1-5 become 0-4 because the neural network uses
    zero-based class indices. The returned preprocessor is saved
    so new patient records can be transformed in exactly the
    same way later.
    """
    required_columns = TRIAGE_FEATURES + [TRIAGE_TARGET]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "The input data is missing required columns: "
            + ", ".join(missing_columns)
        )

    X = data[TRIAGE_FEATURES].copy()
    y = data[TRIAGE_TARGET].astype(int).copy()

    class_counts = y.value_counts()
    if len(class_counts) < len(KTAS_CLASSES):
        raise ValueError(
            "The dataset must contain all five KTAS classes to train "
            "and evaluate this five-class model."
        )
    if class_counts.min() < 2:
        raise ValueError(
            "Each KTAS class needs at least two records for a stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_deep_learning_preprocessor()

    # Here I am fitting preprocessing on training records only.
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    # Here I am converting KTAS labels from 1-5 to 0-4.
    y_train = y_train.to_numpy(dtype=np.int64) - 1
    y_test = y_test.to_numpy(dtype=np.int64) - 1

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except (AttributeError, ValueError):
        feature_names = [f"feature_{index}" for index in range(X_train.shape[1])]

    print("\n============================================")
    print("DEEP LEARNING DATA PREPARATION")
    print("============================================")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Input features after encoding: {X_train.shape[1]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    )


# ============================================================
# BUILD DEEP LEARNING MODEL
# ============================================================

def build_deep_learning_model(input_size, number_of_classes=5):
    """
    Here I am creating a feed-forward neural network.

    Architecture:
        Input
        Dense 64 (ReLU)
        Dropout 0.20
        Dense 32 (ReLU)
        Dropout 0.20
        Dense 5 (Softmax)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_size,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(number_of_classes, activation="softmax"),
        ],
        name="predict_care_deep_learning_triage",
    )

    # Here I am configuring the learning algorithm and loss function.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_deep_learning_model(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
):
    """Here I am training the neural network with class balancing."""
    model = build_deep_learning_model(
        input_size=X_train.shape[1],
        number_of_classes=len(KTAS_CLASSES),
    )

    # Here I am giving additional weight to less common KTAS classes.
    class_indices = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=y_train,
    )
    class_weights = {
        int(class_index): float(weight)
        for class_index, weight in zip(class_indices, weights)
    }

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print("\n============================================")
    print("TRAINING DEEP LEARNING MODEL")
    print("============================================")
    print(f"Epoch limit: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Class weights: {class_weights}")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nDeep Learning training completed.")
    return model, history


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_deep_learning_model(model, X_test, y_test):
    """
    Here I am evaluating predictions on the held-out test set.
    I report accuracy, per-class precision/recall/F1, and the
    confusion matrix because accuracy alone can hide poor
    performance on less common KTAS classes.
    """
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=[f"KTAS {level}" for level in KTAS_CLASSES],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3, 4],
    )

    print("\n============================================")
    print("DEEP LEARNING EVALUATION")
    print("============================================")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(matrix)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
        "probabilities": probabilities,
    }


# ============================================================
# SAVE MODEL AND PREPROCESSOR
# ============================================================

def save_deep_learning_model(model, preprocessor, feature_names):
    """
    Here I am saving both the trained neural network and the
    fitted preprocessing pipeline. Saving the fitted pipeline
    means future predictions use the same encoding and scaling.
    """
    create_deep_learning_directories()

    model.save(DEEP_LEARNING_MODEL_FILE)

    preprocessing_data = {
        "preprocessor": preprocessor,
        "feature_names": list(feature_names),
        "feature_columns": list(TRIAGE_FEATURES),
        "target_column": TRIAGE_TARGET,
        "categorical_features": list(TRIAGE_CATEGORICAL_FEATURES),
        "numerical_features": list(TRIAGE_NUMERICAL_FEATURES),
        "ktas_classes": list(KTAS_CLASSES),
    }

    joblib.dump(preprocessing_data, DEEP_LEARNING_PREPROCESSOR_FILE)

    print("\nDeep Learning model saved:")
    print(DEEP_LEARNING_MODEL_FILE)
    print("Deep Learning preprocessor saved:")
    print(DEEP_LEARNING_PREPROCESSOR_FILE)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_deep_learning_model():
    """
    Here I am loading the already trained model and its fitted
    preprocessor. Run the training process once before using
    this function for inference.
    """
    if not os.path.exists(DEEP_LEARNING_MODEL_FILE):
        raise FileNotFoundError(
            "Deep Learning model does not exist yet. "
            "Run this script's training process first."
        )

    if not os.path.exists(DEEP_LEARNING_PREPROCESSOR_FILE):
        raise FileNotFoundError(
            "Deep Learning preprocessor does not exist yet. "
            "Run this script's training process first."
        )

    model = keras.models.load_model(DEEP_LEARNING_MODEL_FILE)
    preprocessing_data = joblib.load(DEEP_LEARNING_PREPROCESSOR_FILE)

    if "preprocessor" not in preprocessing_data:
        raise ValueError(
            "The saved preprocessing file uses an older format. "
            "Retrain this Deep Learning model using the updated script."
        )

    return (
        model,
        preprocessing_data["preprocessor"],
        preprocessing_data["feature_names"],
    )


# ============================================================
# PREDICT KTAS FOR NEW PATIENT DATA
# ============================================================

def predict_ktas(model, preprocessor, patient_data):
    """
    Here I am predicting KTAS for new patient information.

    patient_data must be a pandas DataFrame containing the same
    raw feature columns used for training. A dictionary is also
    accepted and is treated as one patient record.

    I apply the saved preprocessor before passing values to the
    neural network, then convert the output back to KTAS 1-5.
    """
    if isinstance(patient_data, dict):
        patient_data = pd.DataFrame([patient_data])
    elif not isinstance(patient_data, pd.DataFrame):
        raise TypeError(
            "patient_data must be a pandas DataFrame or a dictionary."
        )

    missing_columns = [
        column for column in TRIAGE_FEATURES
        if column not in patient_data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Patient data is missing required features: "
            + ", ".join(missing_columns)
        )

    patient_features = patient_data[TRIAGE_FEATURES].copy()

    for column in TRIAGE_NUMERICAL_FEATURES:
        patient_features[column] = pd.to_numeric(
            patient_features[column], errors="coerce"
        )

    for column in TRIAGE_CATEGORICAL_FEATURES:
        patient_features[column] = patient_features[column].map(
            lambda value: (
                np.nan
                if pd.isna(value) or not str(value).strip()
                else str(value)
            )
        ).astype("object")

    transformed_data = preprocessor.transform(patient_features)
    transformed_data = np.asarray(transformed_data, dtype=np.float32)

    probabilities = model.predict(transformed_data, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)
    predicted_ktas = predicted_classes + 1

    return predicted_ktas, probabilities


# ============================================================
# MODEL STATUS
# ============================================================

def deep_learning_status():
    """Here I am checking whether the model and preprocessor exist."""
    model_exists = os.path.exists(DEEP_LEARNING_MODEL_FILE)
    preprocessor_exists = os.path.exists(DEEP_LEARNING_PREPROCESSOR_FILE)

    print("\n============================================")
    print("DEEP LEARNING STATUS")
    print("============================================")
    print(f"Model exists: {model_exists}")
    print(f"Preprocessor exists: {preprocessor_exists}")

    return {
        "model_exists": model_exists,
        "preprocessor_exists": preprocessor_exists,
    }


# ============================================================
# RUN TRAINING WHEN THIS FILE IS EXECUTED DIRECTLY
# ============================================================

def main():
    """Here I am running the complete Deep Learning training workflow."""
    data = load_triage_dataset()
    data = preprocess_triage_data(data)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    ) = prepare_deep_learning_data(data)

    model, history = train_deep_learning_model(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
    )

    evaluation = evaluate_deep_learning_model(
        model,
        X_test,
        y_test,
    )

    save_deep_learning_model(
        model,
        preprocessor,
        feature_names,
    )

    deep_learning_status()

    print("\n============================================")
    print("DEEP LEARNING RUN SUMMARY")
    print("============================================")
    print(f"Final test accuracy: {evaluation['accuracy']:.4f}")
    print("Saved model and preprocessor are ready for inference.")
    print(
        "This model is an academic prototype and must not be used "
        "to make clinical decisions."
    )

    return model, history, evaluation


if __name__ == "__main__":
    main()
