"""
Predict-Care AI
Deep Learning Component

Here I am implementing the Deep Learning component of my
Predict-Care AI healthcare decision-support system.

This component uses a neural network to predict the
expert KTAS triage level of an emergency patient.

My traditional Machine Learning model is implemented
separately in healthcare_ai.py using Random Forest.

The purpose of this file is to provide a separate
Deep Learning approach for the project.

NOTE: The comments will be deleted in the final version of the code. They are here for
debugging purposes.
"""

import os

# Here I am selecting PyTorch as the backend for Keras.
# This must be done before importing Keras.
os.environ["KERAS_BACKEND"] = "torch"

import joblib
import numpy as np
import pandas as pd
import keras

from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_DIR = "Data"
MODEL_DIR = "Models"

TRIAGE_DATASET = os.path.join(
    DATA_DIR,
    "ktas_cleaned.csv"
)

DEEP_LEARNING_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "deep_learning_triage_model.keras"
)

DEEP_LEARNING_PREPROCESSOR_FILE = os.path.join(
    MODEL_DIR,
    "deep_learning_triage_preprocessor.pkl"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

def create_deep_learning_directories():
    """
    Here I am making sure that the Models directory exists
    before I save my Deep Learning model.
    """

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_triage_dataset():
    """
    Here I am loading the emergency triage dataset.

    """

    if not os.path.exists(TRIAGE_DATASET):
        raise FileNotFoundError(
            f"Dataset not found: {TRIAGE_DATASET}"
        )

    data = pd.read_csv(
        TRIAGE_DATASET,sep = ','
    )

    print("\n============================================")
    print("DEEP LEARNING DATASET")
    print("============================================")

    print(
        f"Dataset shape: {data.shape}"
    )

    return data


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_triage_data(data):
    """
    Here I am selecting the patient information that will
    be used by my Deep Learning model.

    I am predicting KTAS_expert.

    I am not using KTAS_RN, KTAS_expert, mistriage,
    Error_group or other post-triage information as
    input features because that could cause data leakage.
    """

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

    target_column = "KTAS_expert"

    required_columns = feature_columns + [
        target_column
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    data = data[
        required_columns
    ].copy()

    numerical_columns = [
        "Age",
        "NRS_pain",
        "SBP",
        "DBP",
        "HR",
        "RR",
        "BT",
        "Saturation"
    ]

    categorical_columns = [
        "Arrival mode",
        "Injury",
        "Mental",
        "Pain"
    ]

    # Here I am converting numerical fields into numbers.
    for column in numerical_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Here I am converting the target into a number.
    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    # Here I am removing infinite values.
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Here I am removing incomplete records.
    data = data.dropna()

    # Here I am converting categorical fields to text.
    for column in categorical_columns:

        data[column] = data[column].astype(str)

    # Here I am converting the target into integer classes.
    data[target_column] = data[
        target_column
    ].astype(int)

    # Here I am keeping only valid KTAS classes.
    data = data[
        data[target_column].between(1, 5)
    ]

    print(
        f"Records after preprocessing: {len(data)}"
    )

    return data


# ============================================================
# PREPARE DATA FOR NEURAL NETWORK
# ============================================================

def prepare_deep_learning_data(
    data,
    test_size=0.20,
    random_state=42
):
    """
    Here I am converting the patient information into
    numerical data that the neural network can process.

    Categorical variables are one-hot encoded.

    Numerical variables are standardised.

    KTAS classes 1-5 are converted to 0-4 because the
    neural network uses zero-based class labels.
    """

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

    target_column = "KTAS_expert"

    numerical_columns = [
        "Age",
        "NRS_pain",
        "SBP",
        "DBP",
        "HR",
        "RR",
        "BT",
        "Saturation"
    ]

    categorical_columns = [
        "Arrival mode",
        "Injury",
        "Mental",
        "Pain"
    ]

    X = data[
        feature_columns
    ].copy()

    y = data[
        target_column
    ].copy()

    # Here I am converting categorical data into
    # numerical one-hot encoded columns.
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=float
    )

    X = X.astype(float)

    feature_names = X.columns.tolist()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Here I am scaling only the numerical columns.
    scaler = StandardScaler()

    X_train[numerical_columns] = (
        scaler.fit_transform(
            X_train[numerical_columns]
        )
    )

    X_test[numerical_columns] = (
        scaler.transform(
            X_test[numerical_columns]
        )
    )

    # Here I am converting the data into NumPy arrays.
    X_train = X_train.to_numpy(
        dtype=np.float32
    )

    X_test = X_test.to_numpy(
        dtype=np.float32
    )

    y_train = y_train.to_numpy(
        dtype=np.int64
    )

    y_test = y_test.to_numpy(
        dtype=np.int64
    )

    # Here I am changing KTAS 1-5 into classes 0-4.
    y_train = y_train - 1
    y_test = y_test - 1

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    )


# ============================================================
# BUILD DEEP LEARNING MODEL
# ============================================================

def build_deep_learning_model(
    input_size,
    number_of_classes=5
):
    """
    Here I am creating my neural network.

    Architecture:

    Input
       |
    Dense 64 - ReLU
       |
    Dropout
       |
    Dense 32 - ReLU
       |
    Dropout
       |
    Dense 5 - Softmax
       |
    KTAS prediction
    """

    model = keras.Sequential(
        [
            layers.Input(
                shape=(input_size,)
            ),

            layers.Dense(
                64,
                activation="relu"
            ),

            layers.Dropout(
                0.20
            ),

            layers.Dense(
                32,
                activation="relu"
            ),

            layers.Dropout(
                0.20
            ),

            layers.Dense(
                number_of_classes,
                activation="softmax"
            )
        ]
    )

    # Here I am configuring how my neural network learns.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_deep_learning_model(
    X_train,
    y_train,
    epochs=30,
    batch_size=32
):
    """
    Here I am training my Deep Learning neural network.
    """

    model = build_deep_learning_model(
        input_size=X_train.shape[1],
        number_of_classes=5
    )

    print("\n============================================")
    print("TRAINING DEEP LEARNING MODEL")
    print("============================================")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    print(
        "\nDeep Learning training completed."
    )

    return model, history


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_deep_learning_model(
    model,
    X_test,
    y_test
):
    """
    Here I am evaluating the Deep Learning model.

    I am calculating:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix
    """

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[
            0,
            1,
            2,
            3,
            4
        ],
        target_names=[
            "KTAS 1",
            "KTAS 2",
            "KTAS 3",
            "KTAS 4",
            "KTAS 5"
        ],
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            0,
            1,
            2,
            3,
            4
        ]
    )

    print("\n============================================")
    print("DEEP LEARNING EVALUATION")
    print("============================================")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predictions
    }


# ============================================================
# SAVE MODEL
# ============================================================

def save_deep_learning_model(
    model,
    scaler,
    feature_names
):
    """
    Here I am saving the trained neural network and
    preprocessing information.

    This allows me to use the trained model later
    without training it again.
    """

    create_deep_learning_directories()

    model.save(
        DEEP_LEARNING_MODEL_FILE
    )

    preprocessing_data = {
        "scaler": scaler,
        "feature_names": feature_names
    }

    joblib.dump(
        preprocessing_data,
        DEEP_LEARNING_PREPROCESSOR_FILE
    )

    print("\nDeep Learning model saved:")
    print(
        DEEP_LEARNING_MODEL_FILE
    )

    print(
        "Deep Learning preprocessing saved:"
    )

    print(
        DEEP_LEARNING_PREPROCESSOR_FILE
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_deep_learning_model():
    """
    ============================================================
    USE THIS SECTION ONLY AFTER THE MODEL HAS BEEN CREATED
    AND SAVED SUCCESSFULLY.
    ============================================================

    Here I am loading an already trained Deep Learning model.

    I do not need to train the model again when using this
    function.
    """

    if not os.path.exists(
        DEEP_LEARNING_MODEL_FILE
    ):
        raise FileNotFoundError(
            "Deep Learning model does not exist yet. "
            "Run the training process first."
        )

    if not os.path.exists(
        DEEP_LEARNING_PREPROCESSOR_FILE
    ):
        raise FileNotFoundError(
            "Deep Learning preprocessor does not exist yet. "
            "Run the training process first."
        )

    model = keras.models.load_model(
        DEEP_LEARNING_MODEL_FILE
    )

    preprocessing_data = joblib.load(
        DEEP_LEARNING_PREPROCESSOR_FILE
    )

    return (
        model,
        preprocessing_data["scaler"],
        preprocessing_data["feature_names"]
    )


# ============================================================
# PREDICT WITH TRAINED MODEL
# ============================================================

def predict_ktas(
    model,
    patient_data
):
    """
    ============================================================
    USE THIS FUNCTION ONLY AFTER A TRAINED MODEL EXISTS.
    ============================================================

    Here I am using an already trained neural network
    to predict the KTAS level for new patient data.

    patient_data must already be prepared in the same
    feature format used during training.
    """

    probabilities = model.predict(
        patient_data,
        verbose=0
    )

    predicted_class = np.argmax(
        probabilities,
        axis=1
    )

    # Here I am converting the zero-based class back
    # to the original KTAS 1-5 scale.
    predicted_ktas = predicted_class + 1

    return predicted_ktas, probabilities


# ============================================================
# STATUS
# ============================================================

def deep_learning_status():
    """
    Here I am checking whether my Deep Learning model
    has already been created.
    """

    model_exists = os.path.exists(
        DEEP_LEARNING_MODEL_FILE
    )

    preprocessor_exists = os.path.exists(
        DEEP_LEARNING_PREPROCESSOR_FILE
    )

    print("\n============================================")
    print("DEEP LEARNING STATUS")
    print("============================================")

    print(
        f"Model exists: {model_exists}"
    )

    print(
        f"Preprocessor exists: {preprocessor_exists}"
    )

    return {
        "model_exists": model_exists,
        "preprocessor_exists": preprocessor_exists
    }