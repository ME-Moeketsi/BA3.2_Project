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

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ============================================================
# IMPORT PROJECT COMPONENTS
# ============================================================

from healthcare_ai import (
    load_triage_model,
    load_patient_flow_model,
)

from old.nlp_pipeline import predict_ktas as predict_nlp_ktas

from deep_learning import (
    load_deep_learning_model,
    predict_ktas as predict_deep_learning_ktas,
)


# ============================================================
# PROJECT DATA PATHS (case-insensitive folder resolution)
# ============================================================

def _resolve_data_dir():
    """
    Here I look for the project's data folder regardless of whether
    the folder is named 'data' or 'Data'. healthcare_ai.py uses
    'data' while other parts of the project use 'Data', so I make
    sure both work.
    """
    for candidate in ("Data", "data"):
        candidate_path = os.path.join(PROJECT_DIR, candidate)
        if os.path.isdir(candidate_path):
            return candidate_path
    return os.path.join(PROJECT_DIR, "Data")


DATA_DIR = _resolve_data_dir()

TRIAGE_DATASET = os.path.join(DATA_DIR, "ktas_cleaned.csv")
PATIENT_FLOW_DATASET = os.path.join(DATA_DIR, "patient_flow_cleaned.csv")


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
            "KTAS dataset not found: " + TRIAGE_DATASET
        )

    data = pd.read_csv(TRIAGE_DATASET)

    required_columns = [
        "Arrival mode",
        "Injury",
        "Mental",
        "Pain",
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
        values = sorted(values.unique())
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

    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")

    while True:
        try:
            selection = int(input("\nSelect an option: "))

            if 1 <= selection <= len(options):
                return options[selection - 1]

            print(
                f"Please select a number "
                f"between 1 and {len(options)}."
            )

        except ValueError:
            print("Please enter the number of one of the options.")


# ============================================================
# INTEGER INPUT
# ============================================================

def get_integer_input(prompt, minimum=None, maximum=None):
    """
    Here I am collecting an integer and checking
    that it is within the required range.
    """

    while True:
        try:
            value = int(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Please enter a value of at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Please enter a value no greater than {maximum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid whole number.")


# ============================================================
# FLOAT INPUT
# ============================================================

def get_float_input(prompt, minimum=None, maximum=None):
    """
    Here I am collecting a numerical value and
    checking that it is within the required range.
    """

    while True:
        try:
            value = float(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Please enter a value of at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Please enter a value no greater than {maximum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


# ============================================================
# TEXT INPUT
# ============================================================

def get_text_input(prompt):
    """Here I am collecting required text input."""

    while True:
        value = input(prompt).strip()

        if value:
            return value

        print("This field cannot be empty.")


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

    print("\nGROUP")
    print("Enter the code used by the training dataset (1 or 2).")
    patient_data["Group"] = get_integer_input(
        "Group code: ",
        minimum=1,
        maximum=2,
    )

    print("\nSEX")
    print("Enter the code used by the training dataset (1 or 2).")
    patient_data["Sex"] = get_integer_input(
        "Sex code: ",
        minimum=1,
        maximum=2,
    )

    print("\nPATIENT FLOW")
    patient_data["Patients number per hour"] = get_integer_input(
        "Number of patients arriving per hour: ",
        minimum=1,
    )

    patient_data["Age"] = get_integer_input(
        "\nAge: ",
        minimum=0,
        maximum=120,
    )

    patient_data["Arrival mode"] = select_from_menu(
        "ARRIVAL MODE", options["Arrival mode"]
    )
    patient_data["Injury"] = select_from_menu(
        "INJURY", options["Injury"]
    )
    patient_data["Mental"] = select_from_menu(
        "MENTAL STATUS", options["Mental"]
    )
    patient_data["Pain"] = select_from_menu(
        "PAIN", options["Pain"]
    )

    print("\n")
    print("============================================")
    print("PAIN SCORE")
    print("============================================")
    print("Enter the patient's pain level from 0 to 10.")

    patient_data["NRS_pain"] = get_float_input(
        "Pain level (0-10): ",
        minimum=0,
        maximum=10,
    )

    print("\n")
    print("============================================")
    print("BLOOD PRESSURE")
    print("============================================")

    patient_data["SBP"] = get_float_input(
        "Systolic BP (SBP): ", minimum=0
    )
    patient_data["DBP"] = get_float_input(
        "Diastolic BP (DBP): ", minimum=0
    )

    print("\n")
    print("============================================")
    print("HEART RATE")
    print("============================================")

    patient_data["HR"] = get_float_input(
        "Heart rate (beats per minute): ", minimum=0
    )

    print("\n")
    print("============================================")
    print("RESPIRATORY RATE")
    print("============================================")

    patient_data["RR"] = get_float_input(
        "Respiratory rate (breaths per minute): ", minimum=0
    )

    print("\n")
    print("============================================")
    print("BODY TEMPERATURE")
    print("============================================")

    patient_data["BT"] = get_float_input(
        "Body temperature (°C): ", minimum=0
    )

    print("\n")
    print("============================================")
    print("OXYGEN SATURATION")
    print("============================================")

    patient_data["Saturation"] = get_float_input(
        "Oxygen saturation (%): ", minimum=0, maximum=100
    )

    print("\n")
    print("============================================")
    print("CHIEF COMPLAINT")
    print("============================================")
    print("Describe the main reason for the patient's visit.")

    patient_data["Chief_complain"] = get_text_input(
        "Chief complaint: "
    )

    return patient_data


# ============================================================
# TRADITIONAL ML PREDICTION
# ============================================================

def run_traditional_prediction(patient_data):
    """Here I preprocess the complete patient record before predicting KTAS."""

    package = load_triage_model()
    model = package["model"]
    preprocessor = package.get("preprocessor", package["scaler"])
    feature_columns = package["feature_columns"]

    missing_features = [
        column for column in feature_columns
        if column not in patient_data
    ]

    if missing_features:
        raise ValueError(
            "The triage form is missing required features: "
            + ", ".join(missing_features)
            + ". Complete the form before running the prediction."
        )

    patient_df = pd.DataFrame(
        [{column: patient_data[column] for column in feature_columns}]
    )

    transformed_patient = preprocessor.transform(patient_df)

    prediction = model.predict(transformed_patient)

    return int(prediction[0])


# ============================================================
# NLP PREDICTION
# ============================================================

def run_nlp_prediction(chief_complaint):
    """
    Here I am using the NLP model to predict KTAS
    from the patient's chief complaint.
    """

    return predict_nlp_ktas(chief_complaint)


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

def run_deep_learning_prediction(patient_data):
    """
    Here I am using the saved neural network to predict the
    patient's KTAS level.

    predict_ktas() in deep_learning.py applies the saved
    preprocessor internally, so I pass the raw patient record
    and let the module handle encoding and scaling.
    """

    model, preprocessor, feature_names = load_deep_learning_model()

    prediction, probabilities = predict_deep_learning_ktas(
        model,
        preprocessor,
        patient_data,
    )

    predicted_ktas = int(prediction[0])

    return predicted_ktas, probabilities[0]


# ============================================================
# KTAS -> OPERATIONAL PRIORITY
# ============================================================

def map_ktas_to_priority(ktas):
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
    to determine whether the current scenario represents
    a peak or non-peak period.

    The dataset has no wall-clock DateTime column, so peak
    status is derived from the Scenario label and the
    average IsPeakArrival rate.
    """

    if not os.path.exists(PATIENT_FLOW_DATASET):
        raise FileNotFoundError(
            "Patient-flow dataset not found: "
            + PATIENT_FLOW_DATASET
        )

    data = pd.read_csv(PATIENT_FLOW_DATASET)

    required_columns = ["Scenario", "IsPeakArrival"]
    missing = [
        column for column in required_columns
        if column not in data.columns
    ]
    if missing:
        raise ValueError(
            "Patient-flow dataset is missing required columns: "
            + ", ".join(missing)
        )

    data = data.dropna(subset=["Scenario", "IsPeakArrival"])

    if data.empty:
        return 0, "NON-PEAK"

    most_common_scenario = (
        data["Scenario"]
        .astype(str)
        .value_counts()
        .idxmax()
    )

    scenario_lower = most_common_scenario.lower()
    peak_keywords = ("peak", "120", "360")
    scenario_says_peak = any(
        keyword in scenario_lower for keyword in peak_keywords
    )

    average_peak_rate = float(
        pd.to_numeric(
            data["IsPeakArrival"], errors="coerce"
        ).fillna(0).mean()
    )

    is_peak = int(scenario_says_peak or average_peak_rate >= 0.5)

    if is_peak:
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
            "4 doctors",
        ],
    )
    number_of_doctors = int(number_of_doctors.split()[0])

    peak_limit_option = select_from_menu(
        "PEAK DURATION",
        [
            "120 minutes",
            "240 minutes",
            "360 minutes",
        ],
    )
    peak_limit = int(peak_limit_option.split()[0])

    return number_of_doctors, peak_limit


# ============================================================
# PATIENT FLOW PREDICTION
# ============================================================

def run_patient_flow_prediction(
    priority_level,
    is_peak,
    number_of_doctors,
    peak_limit,
):
    """
    Here I am using the saved patient-flow model to predict
    the doctor's waiting time.

    The saved model was trained on these six features, in this
    exact order:

        Priority, IsPeakArrival, NumDoctors, PeakLimit,
        ArrivalTime, Scenario

    The preprocessor that ships with the model scales the five
    numerical features and one-hot encodes the Scenario
    categorical feature. I therefore pass the raw values and
    let the saved preprocessor perform the transformation.
    """

    package = load_patient_flow_model()

    model = package["model"]
    preprocessor = package.get("preprocessor", package.get("scaler"))
    feature_columns = list(package["feature_columns"])
    numerical_columns = list(package.get("numerical_features", []))
    categorical_columns = list(package.get("categorical_features", []))

    # Here I map the KTAS-derived priority code (1, 2, 3) to
    # the numeric Priority column used by the flow dataset.
    priority_numeric = int(priority_level)

    # Here I pick a Scenario label that exists in the training
    # data. The training pipeline validates Scenario values
    # against a known set, so I use one of the supported labels.
    if is_peak and peak_limit == 120:
        scenario = "Peak_120Min"
    elif is_peak and peak_limit == 360:
        scenario = "Peak_360Min"
    elif is_peak:
        scenario = "Baseline_2Doc_240Peak"
    else:
        scenario = "Baseline_2Doc_240Peak"

    # Here I synthesise a plausible ArrivalTime. The dataset
    # stores ArrivalTime as minutes into the simulation run, not
    # as a wall clock. I place this patient at the midpoint of
    # the peak window so the model sees a representative value.
    arrival_time = max(0, int(peak_limit) // 2)

    record = {
        "Priority": priority_numeric,
        "IsPeakArrival": int(is_peak),
        "NumDoctors": int(number_of_doctors),
        "PeakLimit": int(peak_limit),
        "ArrivalTime": arrival_time,
        "Scenario": scenario,
    }

    missing = [
        column for column in feature_columns
        if column not in record
    ]
    if missing:
        raise ValueError(
            "Patient-flow input is missing expected features: "
            + ", ".join(missing)
        )

    input_df = pd.DataFrame(
        [{column: record[column] for column in feature_columns}]
    )

    transformed_input = preprocessor.transform(input_df)
    predicted_wait = model.predict(transformed_input)

    return float(predicted_wait[0])


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

    print("\nHere I am loading the available patient")
    print("input options directly from the KTAS dataset.")

    options = load_ktas_options()

    patient_data = collect_patient_data(options)

    (
        number_of_doctors,
        peak_limit,
    ) = collect_operational_information()

    print("\n")
    print("Running Traditional Machine Learning...")

    traditional_ktas = run_traditional_prediction(patient_data)

    print("Running Natural Language Processing...")

    nlp_result = run_nlp_prediction(
        patient_data["Chief_complain"]
    )
    nlp_ktas = int(nlp_result["predicted_ktas"])

    print("Running Deep Learning...")

    (
        deep_learning_ktas,
        probabilities,
    ) = run_deep_learning_prediction(patient_data)

    (
        priority_level,
        priority_name,
    ) = map_ktas_to_priority(traditional_ktas)

    print("Checking the current patient-flow period...")

    (
        is_peak,
        peak_status,
    ) = determine_current_peak_status()

    print("Predicting doctor waiting time...")

    predicted_wait = run_patient_flow_prediction(
        priority_level,
        is_peak,
        number_of_doctors,
        peak_limit,
    )

    print("\n")
    print("############################################")
    print("#")
    print("#          PREDICT-CARE AI RESULTS")
    print("#")
    print("############################################")

    print(f"\nTraditional ML KTAS:       {traditional_ktas}")
    print(f"NLP KTAS:                  {nlp_ktas}")
    print(f"Deep Learning KTAS:        {deep_learning_ktas}")

    print(
        f"\nNLP Emergency Category:    "
        f"{nlp_result['emergency_category']}"
    )

    print(f"\nOperational Priority:      {priority_name}")
    print(f"Operational Priority Code: {priority_level}")
    print(f"Current Period:            {peak_status}")
    print(f"Available Doctors:         {number_of_doctors}")
    print(f"Peak Duration:             {peak_limit} minutes")
    print(f"\nPredicted Doctor Wait:     {predicted_wait:.2f} minutes")

    print("\n")
    print("============================================")
    print("MODEL COMPARISON")
    print("============================================")

    print(f"Traditional ML:  KTAS {traditional_ktas}")
    print(f"NLP:             KTAS {nlp_ktas}")
    print(f"Deep Learning:   KTAS {deep_learning_ktas}")

    if (
        traditional_ktas
        == nlp_ktas
        == deep_learning_ktas
    ):
        print("\nAll three models produced the same KTAS level.")
    else:
        print("\nThe models produced different KTAS levels.")
        print("The predictions are being displayed separately.")
        print("They are not being artificially averaged.")

    print("\n")
    print("============================================")
    print("END OF PREDICT-CARE TEST")
    print("============================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()