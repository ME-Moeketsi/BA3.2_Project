"""
Predict-Care AI
Integrated Prediction Test

Here I am testing the main Predict-Care AI prediction
components together using information entered at runtime.

The categorical options are loaded directly from the
KTAS dataset so that I do not have to hardcode medical
or dataset-specific category values.

This test checks:
1. Traditional Machine Learning KTAS prediction
2. NLP KTAS prediction
3. Deep Learning KTAS prediction
4. KTAS to operational priority mapping
5. Peak / non-peak detection
6. Patient-flow doctor waiting-time prediction
"""

import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# Here I am finding the main project directory.
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ============================================================
# IMPORT PROJECT COMPONENTS
# ============================================================

# Here I am importing the existing Traditional ML models.
from healthcare_ai import (
    load_triage_model,
    load_patient_flow_model
)

# Here I am importing the existing NLP prediction pipeline.
from NLP.nlp_pipeline import predict_ktas as predict_nlp_ktas

# Here I am importing the existing Deep Learning functions.
from deep_learning import (
    load_deep_learning_model,
    predict_ktas as predict_deep_learning_ktas
)


# ============================================================
# PROJECT DATA
# ============================================================

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "Data"
)

TRIAGE_DATASET = os.path.join(
    DATA_DIR,
    "ktas_cleaned.csv"
)

PATIENT_FLOW_DATASET = os.path.join(
    DATA_DIR,
    "patient_flow_cleaned.csv"
)


# ============================================================
# DATASET LOADING
# ============================================================

def load_ktas_options():
    """
    Here I am loading the KTAS dataset so that the
    categorical input menus use the actual values found
    in the training data.
    """

    if not os.path.exists(TRIAGE_DATASET):

        raise FileNotFoundError(
            "KTAS dataset not found: "
            + TRIAGE_DATASET
        )

    data = pd.read_csv(
        TRIAGE_DATASET
    )

    required_columns = [
        "Arrival mode",
        "Injury",
        "Mental",
        "Pain"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing categorical columns: "
            + ", ".join(missing_columns)
        )

    options = {}

    for column in required_columns:

        values = (
            data[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = sorted(
            values.unique()
        )

        options[column] = values

    return options


# ============================================================
# MENU INPUT
# ============================================================

def select_from_menu(title, options):
    """
    Here I am displaying numbered options so the user
    can select a value without having to type the
    dataset value manually.
    """

    print("\n")
    print("============================================")
    print(title)
    print("============================================")

    for index, option in enumerate(
        options,
        start=1
    ):

        print(
            f"{index}. {option}"
        )

    while True:

        try:

            selection = int(
                input(
                    "\nSelect an option: "
                )
            )

            if 1 <= selection <= len(options):

                return options[
                    selection - 1
                ]

            print(
                f"Please select a number "
                f"between 1 and {len(options)}."
            )

        except ValueError:

            print(
                "Please enter the number "
                "of one of the options."
            )


# ============================================================
# INTEGER INPUT
# ============================================================

def get_integer_input(
    prompt,
    minimum=None,
    maximum=None
):
    """
    Here I am collecting an integer and checking
    that it is within the required range.
    """

    while True:

        try:

            value = int(
                input(prompt)
            )

            if (
                minimum is not None
                and value < minimum
            ):

                print(
                    f"Please enter a value "
                    f"of at least {minimum}."
                )

                continue

            if (
                maximum is not None
                and value > maximum
            ):

                print(
                    f"Please enter a value "
                    f"no greater than {maximum}."
                )

                continue

            return value

        except ValueError:

            print(
                "Please enter a valid whole number."
            )


# ============================================================
# FLOAT INPUT
# ============================================================

def get_float_input(
    prompt,
    minimum=None,
    maximum=None
):
    """
    Here I am collecting a numerical value and
    checking that it is within the required range.
    """

    while True:

        try:

            value = float(
                input(prompt)
            )

            if (
                minimum is not None
                and value < minimum
            ):

                print(
                    f"Please enter a value "
                    f"of at least {minimum}."
                )

                continue

            if (
                maximum is not None
                and value > maximum
            ):

                print(
                    f"Please enter a value "
                    f"no greater than {maximum}."
                )

                continue

            return value

        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# TEXT INPUT
# ============================================================

def get_text_input(prompt):
    """
    Here I am collecting required text input.
    """

    while True:

        value = input(prompt).strip()

        if value:

            return value

        print(
            "This field cannot be empty."
        )


# ============================================================
# PATIENT INPUT
# ============================================================

def collect_patient_data(options):
    """
    Here I am collecting the patient's clinical
    information using guided menus and numerical inputs.
    """

    print("\n")
    print("############################################")
    print("#")
    print("#       PATIENT CLINICAL INFORMATION")
    print("#")
    print("############################################")

    patient_data = {}

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    patient_data["Age"] = get_integer_input(
        "\nAge: ",
        minimum=0,
        maximum=120
    )

    # --------------------------------------------------------
    # ARRIVAL MODE
    # --------------------------------------------------------

    patient_data["Arrival mode"] = (
        select_from_menu(
            "ARRIVAL MODE",
            options["Arrival mode"]
        )
    )

    # --------------------------------------------------------
    # INJURY
    # --------------------------------------------------------

    patient_data["Injury"] = (
        select_from_menu(
            "INJURY",
            options["Injury"]
        )
    )

    # --------------------------------------------------------
    # MENTAL STATUS
    # --------------------------------------------------------

    patient_data["Mental"] = (
        select_from_menu(
            "MENTAL STATUS",
            options["Mental"]
        )
    )

    # --------------------------------------------------------
    # PAIN
    # --------------------------------------------------------

    patient_data["Pain"] = (
        select_from_menu(
            "PAIN",
            options["Pain"]
        )
    )

    # --------------------------------------------------------
    # PAIN SCORE
    # --------------------------------------------------------

    print("\n")
    print("============================================")
    print("PAIN SCORE")
    print("============================================")

    print(
        "Enter the patient's pain level from 0 to 10."
    )

    patient_data["NRS_pain"] = get_float_input(
        "Pain level (0-10): ",
        minimum=0,
        maximum=10
    )

    # --------------------------------------------------------
    # BLOOD PRESSURE
    # --------------------------------------------------------

    print("\n")
    print("============================================")
    print("BLOOD PRESSURE")
    print("============================================")

    patient_data["SBP"] = get_float_input(
        "Systolic BP (SBP): ",
        minimum=0
    )

    patient_data["DBP"] = get_float_input(
        "Diastolic BP (DBP): ",
        minimum=0
    )

    # --------------------------------------------------------
    # HEART RATE
    # --------------------------------------------------------

    print("\n")
    print("============================================")
    print("HEART RATE")
    print("============================================")

    patient_data["HR"] = get_float_input(
        "Heart rate (beats per minute): ",
        minimum=0
    )

    # --------------------------------------------------------
    # RESPIRATORY RATE
    # --------------------------------------------------------

    print("\n")
    print("============================================")
    print("RESPIRATORY RATE")
    print("============================================")

    patient_data["RR"] = get_float_input(
        "Respiratory rate (breaths per minute): ",
        minimum=0
    )

    # --------------------------------------------------------
    # BODY TEMPERATURE
    # --------------------------------------------------------

    print("\n")
    print("============================================")
    print("BODY TEMPERATURE")
    print("============================================")

    patient_data["BT"] = get_float_input(
        "Body temperature (°C): ",
        minimum=0
    )

    # --------------------------------------------------------
    # OXYGEN SATURATION
    # --------------------------------------------------------

    print("\n")
    print("============================================")
    print("OXYGEN SATURATION")
    print("============================================")

    patient_data["Saturation"] = get_float_input(
        "Oxygen saturation (%): ",
        minimum=0,
        maximum=100
    )

    # --------------------------------------------------------
    # CHIEF COMPLAINT
    # --------------------------------------------------------

    print("\n")
    print("============================================")
    print("CHIEF COMPLAINT")
    print("============================================")

    print(
        "Describe the main reason for the patient's visit."
    )

    patient_data["Chief_complain"] = get_text_input(
        "Chief complaint: "
    )

    return patient_data


# ============================================================
# DEEP LEARNING INPUT PREPARATION
# ============================================================

def prepare_deep_learning_input(
    patient_data,
    feature_names,
    scaler
):
    """
    Here I am converting the runtime patient information
    into the same feature structure used by the trained
    Deep Learning model.
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

    patient_df = pd.DataFrame(
        [patient_data]
    )

    patient_df = patient_df[
        feature_columns
    ].copy()

    for column in numerical_columns:

        patient_df[column] = pd.to_numeric(
            patient_df[column],
            errors="raise"
        )

    for column in categorical_columns:

        patient_df[column] = (
            patient_df[column]
            .astype(str)
        )

    patient_df = pd.get_dummies(
        patient_df,
        columns=categorical_columns,
        dtype=float
    )

    # Here I am making the runtime input match the
    # exact features saved with the trained model.
    patient_df = patient_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    patient_df[numerical_columns] = (
        scaler.transform(
            patient_df[numerical_columns]
        )
    )

    return patient_df.to_numpy(
        dtype=np.float32
    )


# ============================================================
# TRADITIONAL ML PREDICTION
# ============================================================

def run_traditional_prediction(
    patient_data
):
    """
    Here I am using the saved Random Forest triage
    model to predict the patient's KTAS level.
    """

    package = load_triage_model()

    model = package["model"]
    scaler = package["scaler"]
    feature_columns = package["feature_columns"]

    patient_df = pd.DataFrame(
        [patient_data]
    )

    patient_df = patient_df[
        feature_columns
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

    for column in numerical_columns:

        patient_df[column] = pd.to_numeric(
            patient_df[column],
            errors="raise"
        )

    patient_df[numerical_columns] = (
        scaler.transform(
            patient_df[numerical_columns]
        )
    )

    prediction = model.predict(
        patient_df
    )

    return int(
        prediction[0]
    )


# ============================================================
# NLP PREDICTION
# ============================================================

def run_nlp_prediction(
    chief_complaint
):
    """
    Here I am using the NLP model to predict KTAS
    from the patient's chief complaint.
    """

    return predict_nlp_ktas(
        chief_complaint
    )


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

def run_deep_learning_prediction(
    patient_data
):
    """
    Here I am using the saved neural network to
    predict the patient's KTAS level.
    """

    model, scaler, feature_names = (
        load_deep_learning_model()
    )

    prepared_data = (
        prepare_deep_learning_input(
            patient_data,
            feature_names,
            scaler
        )
    )

    prediction, probabilities = (
        predict_deep_learning_ktas(
            model,
            prepared_data
        )
    )

    predicted_ktas = int(
        prediction[0]
    )

    return (
        predicted_ktas,
        probabilities[0]
    )


# ============================================================
# KTAS → OPERATIONAL PRIORITY
# ============================================================

def map_ktas_to_priority(
    ktas
):
    """
    Here I am applying the Predict-Care integration
    rule between KTAS and the operational priority used
    by the patient-flow simulation.

    KTAS 1 and 2 -> HIGH
    KTAS 3       -> MEDIUM
    KTAS 4 and 5 -> LOW
    """

    if ktas in [1, 2]:

        return 3, "HIGH"

    if ktas == 3:

        return 2, "MEDIUM"

    return 1, "LOW"


# ============================================================
# PEAK STATUS
# ============================================================

def determine_current_peak_status():
    """
    Here I am using the historical patient-flow dataset
    to determine whether the current hour normally
    represents a peak or non-peak period.

    I am not hardcoding specific clock hours.
    """

    if not os.path.exists(
        PATIENT_FLOW_DATASET
    ):

        raise FileNotFoundError(
            "Patient-flow dataset not found: "
            + PATIENT_FLOW_DATASET
        )

    data = pd.read_csv(
        PATIENT_FLOW_DATASET
    )

    data["DateTime"] = pd.to_datetime(
        data["DateTime"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "DateTime",
            "IsPeakArrival"
        ]
    )

    data["Hour"] = (
        data["DateTime"].dt.hour
    )

    current_hour = datetime.now().hour

    hourly_peak_rate = (
        data.groupby("Hour")[
            "IsPeakArrival"
        ].mean()
    )

    if current_hour not in hourly_peak_rate.index:

        return 0, "NON-PEAK"

    peak_rate = hourly_peak_rate[
        current_hour
    ]

    if peak_rate >= 0.5:

        return 1, "PEAK"

    return 0, "NON-PEAK"


# ============================================================
# OPERATIONAL INPUT
# ============================================================

def collect_operational_information():
    """
    Here I am collecting the information needed by
    the patient-flow model.
    """

    print("\n")
    print("############################################")
    print("#")
    print("#       OPERATIONAL INFORMATION")
    print("#")
    print("############################################")

    number_of_doctors = select_from_menu(
        "NUMBER OF AVAILABLE DOCTORS",
        [
            "1 doctor",
            "2 doctors",
            "3 doctors",
            "4 doctors"
        ]
    )

    number_of_doctors = int(
        number_of_doctors.split()[0]
    )

    peak_limit_option = select_from_menu(
        "PEAK DURATION",
        [
            "120 minutes",
            "240 minutes",
            "360 minutes"
        ]
    )

    peak_limit = int(
        peak_limit_option.split()[0]
    )

    return (
        number_of_doctors,
        peak_limit
    )


# ============================================================
# PATIENT FLOW PREDICTION
# ============================================================

def run_patient_flow_prediction(
    priority_level,
    is_peak,
    number_of_doctors,
    peak_limit
):
    """
    Here I am using the saved patient-flow model
    to predict the doctor's waiting time.
    """

    package = load_patient_flow_model()

    model = package["model"]
    scaler = package["scaler"]
    feature_columns = package["feature_columns"]

    patient_flow_input = pd.DataFrame(
        [[
            priority_level,
            is_peak,
            number_of_doctors,
            peak_limit
        ]],
        columns=feature_columns
    )

    numerical_columns = [
        "priority_level",
        "IsPeakArrival",
        "num_doctors",
        "PeakLimit"
    ]

    patient_flow_input[numerical_columns] = (
        scaler.transform(
            patient_flow_input[
                numerical_columns
            ]
        )
    )

    predicted_wait = model.predict(
        patient_flow_input
    )

    return float(
        predicted_wait[0]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("############################################")
    print("#")
    print("#          PREDICT-CARE AI")
    print("#       INTEGRATED PREDICTION TEST")
    print("#")
    print("############################################")

    print(
        "\nHere I am loading the available patient"
    )

    print(
        "input options directly from the KTAS dataset."
    )

    # --------------------------------------------------------
    # LOAD DATASET OPTIONS
    # --------------------------------------------------------

    options = load_ktas_options()

    # --------------------------------------------------------
    # COLLECT PATIENT INFORMATION
    # --------------------------------------------------------

    patient_data = collect_patient_data(
        options
    )

    # --------------------------------------------------------
    # COLLECT OPERATIONAL INFORMATION
    # --------------------------------------------------------

    (
        number_of_doctors,
        peak_limit
    ) = collect_operational_information()

    # --------------------------------------------------------
    # TRADITIONAL ML
    # --------------------------------------------------------

    print("\n")
    print(
        "Running Traditional Machine Learning..."
    )

    traditional_ktas = (
        run_traditional_prediction(
            patient_data
        )
    )

    # --------------------------------------------------------
    # NLP
    # --------------------------------------------------------

    print(
        "Running Natural Language Processing..."
    )

    nlp_result = run_nlp_prediction(
        patient_data[
            "Chief_complain"
        ]
    )

    nlp_ktas = int(
        nlp_result[
            "predicted_ktas"
        ]
    )

    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    print(
        "Running Deep Learning..."
    )

    (
        deep_learning_ktas,
        probabilities
    ) = run_deep_learning_prediction(
        patient_data
    )

    # --------------------------------------------------------
    # OPERATIONAL PRIORITY
    # --------------------------------------------------------

    # Here I am using the Traditional ML KTAS as the
    # primary KTAS for the operational waiting-time
    # prediction.
    (
        priority_level,
        priority_name
    ) = map_ktas_to_priority(
        traditional_ktas
    )

    # --------------------------------------------------------
    # CURRENT PEAK STATUS
    # --------------------------------------------------------

    print(
        "Checking the current patient-flow period..."
    )

    (
        is_peak,
        peak_status
    ) = determine_current_peak_status()

    # --------------------------------------------------------
    # PATIENT FLOW
    # --------------------------------------------------------

    print(
        "Predicting doctor waiting time..."
    )

    predicted_wait = (
        run_patient_flow_prediction(
            priority_level,
            is_peak,
            number_of_doctors,
            peak_limit
        )
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("############################################")
    print("#")
    print("#          PREDICT-CARE AI RESULTS")
    print("#")
    print("############################################")

    print(
        f"\nTraditional ML KTAS:       "
        f"{traditional_ktas}"
    )

    print(
        f"NLP KTAS:                  "
        f"{nlp_ktas}"
    )

    print(
        f"Deep Learning KTAS:        "
        f"{deep_learning_ktas}"
    )

    print(
        f"\nNLP Emergency Category:    "
        f"{nlp_result['emergency_category']}"
    )

    print(
        f"\nOperational Priority:      "
        f"{priority_name}"
    )

    print(
        f"Operational Priority Code: "
        f"{priority_level}"
    )

    print(
        f"Current Period:            "
        f"{peak_status}"
    )

    print(
        f"Available Doctors:         "
        f"{number_of_doctors}"
    )

    print(
        f"Peak Duration:             "
        f"{peak_limit} minutes"
    )

    print(
        f"\nPredicted Doctor Wait:     "
        f"{predicted_wait:.2f} minutes"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print("\n")
    print("============================================")
    print("MODEL COMPARISON")
    print("============================================")

    print(
        f"Traditional ML:  KTAS {traditional_ktas}"
    )

    print(
        f"NLP:             KTAS {nlp_ktas}"
    )

    print(
        f"Deep Learning:   KTAS {deep_learning_ktas}"
    )

    if (
        traditional_ktas
        == nlp_ktas
        == deep_learning_ktas
    ):

        print(
            "\nAll three models produced the same KTAS level."
        )

    else:

        print(
            "\nThe models produced different KTAS levels."
        )

        print(
            "The predictions are being displayed separately."
        )

        print(
            "They are not being artificially averaged."
        )

    print("\n")
    print("============================================")
    print("END OF PREDICT-CARE TEST")
    print("============================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()