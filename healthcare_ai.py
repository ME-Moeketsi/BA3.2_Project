# IMPORTS

# Here I am importing the operating-system module because
# I will use it to create and manage my project directories.
import os

# Here I am importing joblib because I will use it to save
# and load my trained machine-learning model and scaler.
import joblib

# Here I am importing NumPy because I will need it for
# numerical operations during machine-learning processing.
import numpy as np

# Here I am importing pandas because I will use it to
# load, clean, explore, and manipulate my dataset.
import pandas as pd

# Here I am importing train_test_split because I will use it
# to divide my dataset into training and testing data.
from sklearn.model_selection import train_test_split

# Here I am importing StandardScaler because I will use it
# to scale my numerical features before making predictions.
from sklearn.preprocessing import StandardScaler

# Here I am importing classification metrics because I will
# use them to evaluate how well my triage(for now) model performs.
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)

# Here I am importing RandomForestClassifier because I will
# use it as my machine-learning algorithm for predicting
# the patient's KTAS triage category.
from sklearn.ensemble import RandomForestClassifier

# CONFIGURATION

# Here I am defining the directory where I will keep my
# healthcare datasets.
DATA_DIR = "Data"

# Here I am defining the directory where I will save
# my trained machine-learning models.
MODEL_DIR = "Models"

# Here I am defining the location of my emergency triage
# dataset.
DATASET_FILE = os.path.join(DATA_DIR, "data.csv")

# Here I am defining the location where I will save
# my trained triage classification model.
TRIAGE_MODEL_FILE = os.path.join(MODEL_DIR,"triage_model.pkl")

# Here I am defining the location where I will save
# my feature scaler.
TRIAGE_SCALER_FILE = os.path.join(MODEL_DIR,"triage_scaler.pkl")

# DIRECTORY MANAGEMENT

def create_directories():
    """
    Here I am creating the directories that I need
    for my dataset and trained model files.
    """

    # Here I am creating the data directory if it
    # does not already exist.
    os.makedirs(DATA_DIR, exist_ok=True)

    # Here I am creating the models directory if it
    # does not already exist.
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Project directories are ready.")

# DATA LOADING

def load_dataset(file_path=DATASET_FILE):
    """
    Here I am loading my emergency triage dataset.

    The Kaggle dataset uses a semicolon as its separator
    and uses cp1254 encoding.
    """

    # Here I am loading my CSV file into a pandas DataFrame.
    data = pd.read_csv(
        file_path,
        sep=";",
        encoding="cp1254"
    )

    print("\nDataset loaded successfully.")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    return data

# DATASET EXPLORATION

def explore_dataset(data):
    """
    Here I am exploring my dataset so that I can understand
    its structure, columns, data types, missing values,
    and basic statistics before training my model.
    """

    print("\n===== DATASET COLUMNS =====")
    print(data.columns.tolist())

    print("\n===== FIRST 5 RECORDS =====")
    print(data.head())

    print("\n===== DATASET INFORMATION =====")
    print(data.info())

    print("\n===== MISSING VALUES =====")
    print(data.isnull().sum())

    print("\n===== DATASET STATISTICS =====")
    print(data.describe(include="all"))

# DATA PREPROCESSING


def preprocess_data(data):
    """
    Here I am cleaning and preparing my dataset before
    I use it for machine learning.
    """

    # Here I am creating a copy so that I do not
    # accidentally modify my original dataset.
    data = data.copy()

    # Here I am removing duplicate records because
    # duplicate patient records could affect my model.
    data = data.drop_duplicates()

    # Here I am checking numeric columns and filling
    # missing numeric values with their median values.
    numeric_columns = data.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())

    print("\nData preprocessing completed.")
    print(f"Rows after preprocessing: {len(data)}")

    return data

# PREPARE DATA FOR MACHINE LEARNING

def prepare_triage_data(data, target_column="KTAS_expert", test_size=0.2):
    """
    Here I am preparing my emergency triage data for
    classification.

    My target is KTAS_expert because I want my model
    to predict the triage category.
    """

    # Here I am checking that my target column exists
    # before I continue with model preparation.
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    # Here I am defining the features that I want
    # my model to use for triage prediction.
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
        "BT"
    ]

    # Here I am checking that all my selected features
    # exist in the dataset.
    missing_features = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    # Here I am selecting my input features.
    X = data[feature_columns].copy()

    # Here I am selecting KTAS_expert as my target.
    y = data[target_column].copy()

    # Here I am converting categorical feature values
    # into numerical values so that my machine-learning
    # algorithm can process them.
    X = pd.get_dummies(
        X,
        columns=[
            "Arrival mode",
            "Injury",
            "Mental",
            "Pain"
        ],
        drop_first=False
    )

    # Here I am splitting my data into training and
    # testing portions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Here I am creating a scaler for my numerical features.
    scaler = StandardScaler()

    # Here I am scaling my training data.
    X_train = scaler.fit_transform(X_train)

    # Here I am using the same scaler to transform
    # my testing data.
    X_test = scaler.transform(X_test)

    print("\nTriage data preparation completed.")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )
    
# TRAIN TRIAGE CLASSIFIER


def train_triage_model(X_train, y_train):
    """
    Here I am training my Random Forest classification
    model to predict the KTAS triage category.
    """

    # Here I am creating my Random Forest classifier.
    model = RandomForestClassifier(n_estimators=100,random_state=42)

    # Here I am training my model using my training data.
    model.fit(X_train, y_train)

    print("\nTriage classification model trained.")

    return model

# EVALUATE TRIAGE MODEL

def evaluate_triage_model(model, X_test, y_test):
    """
    Here I am evaluating my triage model to determine
    how accurately it predicts the KTAS category.
    """

    # Here I am using my trained model to predict
    # the KTAS categories for my test data.
    predictions = model.predict(X_test)

    # Here I am calculating the overall accuracy
    # of my classification model.
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n===== TRIAGE MODEL EVALUATION =====")
    print(f"Accuracy: {accuracy:.4f}")

    # Here I am displaying precision, recall, and F1-score
    # for each KTAS category.
    print("\nClassification Report:")
    print(
        classification_report(y_test,predictions)
    )

    # Here I am displaying the confusion matrix so that
    # I can see how my model is confusing different
    # KTAS categories.
    print("\nConfusion Matrix:")
    print(
        confusion_matrix(y_test,predictions)
    )

    return predictions

# MAKE TRIAGE PREDICTION

def make_triage_prediction(
    model,
    scaler,
    input_data,
    feature_columns
):
    """
    Here I am using my trained model to predict the
    KTAS category for new patient information.
    """

    # Here I am converting the new patient information
    # into a pandas DataFrame.
    input_df = pd.DataFrame(
        [input_data]
    )

    # Here I am converting categorical values into the
    # same numerical format used during model training.
    input_df = pd.get_dummies(
        input_df,
        columns=[
            "Arrival mode",
            "Injury",
            "Mental",
            "Pain"
        ],
        drop_first=False
    )

    # Here I am making sure the new patient data has
    # the same feature structure expected by my model.
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Here I am scaling the new patient information
    # using the scaler that I fitted during training.
    input_scaled = scaler.transform(
        input_df
    )

    # Here I am asking my trained model to predict
    # the patient's KTAS category.
    prediction = model.predict(
        input_scaled
    )

    return prediction[0]

# SAVE MODEL

def save_triage_model(
    model,
    scaler
):
    """
    Here I am saving my trained triage model and scaler
    so that I can use them later without retraining.
    """

    # Here I am saving my trained classification model.
    joblib.dump(model,TRIAGE_MODEL_FILE)

    # Here I am saving my feature scaler.
    joblib.dump(scaler,TRIAGE_SCALER_FILE)

    print("\nTriage model saved successfully.")
    print(f"Model: {TRIAGE_MODEL_FILE}")
    print(f"Scaler: {TRIAGE_SCALER_FILE}")

# LOAD MODEL

def load_triage_model():
    """
    Here I am loading my previously trained triage model
    and scaler so that I can use them for predictions.
    """

    # Here I am loading my saved Random Forest model.
    model = joblib.load(TRIAGE_MODEL_FILE)

    # Here I am loading my saved feature scaler.
    scaler = joblib.load(TRIAGE_SCALER_FILE)

    print("\nTriage model loaded successfully.")

    return model, scaler

# SYSTEM STATUS

def system_status():
    """
    Here I am displaying the current status of the
    Predict-Care AI machine-learning component.
    """
    
    print("\n========================PREDICT-CARE AI=========================")
    print("Healthcare ML Component")
    print("-----------------------------------")
    print("Triage Classification: READY")
    print("Random Forest Classifier: READY")
    print("Model Evaluation: READY")
    print("Prediction: READY")
    print("Model Saving/Loading: READY")
    print("-----------------------------------")
    print("Hospital Simulation Model: NOT ADDED YET")
    print("===================================")
    
    ## The rest will come here however i am gonna make them a separate module for now because this is already 
    # getting too long and i want to keep it clean and organized.