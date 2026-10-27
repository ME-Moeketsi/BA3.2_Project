"""
Predict-Care AI
Integrated Prediction Test

Same logic as the GUI's pipeline, but driven from the terminal.
"""

import os
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


from healthcare_ai import (
    load_triage_model,
    load_patient_flow_model,
)

from NLP.pipeline import predict_ktas_proba as predict_nlp_proba

from deep_learning import (
    load_deep_learning_model,
    predict_ktas as predict_deep_learning_ktas,
)

from GUI import gui_logic as logic


MODEL_WEIGHTS = logic.MODEL_WEIGHTS


def _resolve_data_dir():
    for candidate in ("Data", "data"):
        candidate_path = os.path.join(PROJECT_DIR, candidate)
        if os.path.isdir(candidate_path):
            return candidate_path
    return os.path.join(PROJECT_DIR, "Data")


DATA_DIR = _resolve_data_dir()
TRIAGE_DATASET = os.path.join(DATA_DIR, "ktas_cleaned.csv")
PATIENT_FLOW_DATASET = os.path.join(DATA_DIR, "patient_flow_cleaned.csv")


def load_ktas_options():
    if not os.path.exists(TRIAGE_DATASET):
        raise FileNotFoundError(
            "KTAS dataset not found: " + TRIAGE_DATASET
        )
    data = pd.read_csv(TRIAGE_DATASET)
    required = ["Arrival mode", "Injury", "Mental", "Pain"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing categorical columns: " + ", ".join(missing))
    options = {}
    for column in required:
        values = data[column].dropna().astype(str).str.strip().unique()
        options[column] = sorted(values)
    return options


def select_from_menu(title, options, column_name=None):
    print("\n")
    print("============================================")
    print(title)
    print("============================================")
    for index, option in enumerate(options, start=1):
        if column_name:
            print(str(index) + ". "
                  + logic.display_options_for(column_name, [option])[0])
        else:
            print(str(index) + ". " + str(option))
    while True:
        try:
            selection = int(input("\nSelect an option: "))
            if 1 <= selection <= len(options):
                return options[selection - 1]
            print("Please select a number between 1 and "
                  + str(len(options)) + ".")
        except ValueError:
            print("Please enter the number of one of the options.")


def get_integer_input(prompt, minimum=None, maximum=None):
    while True:
        try:
            value = int(input(prompt))
            if minimum is not None and value < minimum:
                print("Please enter a value of at least "
                      + str(minimum) + ".")
                continue
            if maximum is not None and value > maximum:
                print("Please enter a value no greater than "
                      + str(maximum) + ".")
                continue
            return value
        except ValueError:
            print("Please enter a valid whole number.")


def get_float_input(prompt, minimum=None, maximum=None):
    while True:
        try:
            value = float(input(prompt))
            if minimum is not None and value < minimum:
                print("Please enter a value of at least "
                      + str(minimum) + ".")
                continue
            if maximum is not None and value > maximum:
                print("Please enter a value no greater than "
                      + str(maximum) + ".")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def get_text_input(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This field cannot be empty.")


def collect_patient_data(options):
    print("\n")
    print("############################################")
    print("#       PATIENT CLINICAL INFORMATION")
    print("############################################")

    patient_data = {}

    print("\nGROUP")
    patient_data["Group"] = get_integer_input(
        "Group code (1-2): ", minimum=1, maximum=2
    )

    print("\nSEX")
    patient_data["Sex"] = get_integer_input(
        "Sex code (1-2): ", minimum=1, maximum=2
    )

    print("\nPATIENT FLOW")
    patient_data["Patients number per hour"] = get_integer_input(
        "Number of patients arriving per hour: ", minimum=1
    )

    patient_data["Age"] = get_integer_input(
        "\nAge: ", minimum=0, maximum=120
    )

    patient_data["Arrival mode"] = select_from_menu(
        "ARRIVAL MODE", options["Arrival mode"],
        column_name="Arrival mode",
    )
    patient_data["Injury"] = select_from_menu(
        "INJURY", options["Injury"], column_name="Injury"
    )
    patient_data["Mental"] = select_from_menu(
        "MENTAL STATUS", options["Mental"], column_name="Mental"
    )
    patient_data["Pain"] = select_from_menu(
        "PAIN", options["Pain"], column_name="Pain"
    )

    print("\n============================================")
    print("PAIN SCORE")
    print("============================================")
    patient_data["NRS_pain"] = get_float_input(
        "Pain level (0-10): ", minimum=0, maximum=10
    )

    print("\n============================================")
    print("BLOOD PRESSURE")
    print("============================================")
    patient_data["SBP"] = get_float_input("Systolic BP (SBP): ", minimum=0)
    patient_data["DBP"] = get_float_input("Diastolic BP (DBP): ", minimum=0)

    print("\n============================================")
    print("HEART RATE")
    print("============================================")
    patient_data["HR"] = get_float_input(
        "Heart rate: ", minimum=0
    )

    print("\n============================================")
    print("RESPIRATORY RATE")
    print("============================================")
    patient_data["RR"] = get_float_input(
        "Respiratory rate: ", minimum=0
    )

    print("\n============================================")
    print("BODY TEMPERATURE")
    print("============================================")
    patient_data["BT"] = get_float_input(
        "Body temperature (C): ", minimum=0
    )

    print("\n============================================")
    print("OXYGEN SATURATION")
    print("============================================")
    patient_data["Saturation"] = get_float_input(
        "Oxygen saturation (%): ", minimum=0, maximum=100
    )

    print("\n============================================")
    print("CHIEF COMPLAINT")
    print("============================================")
    patient_data["Chief_complain"] = get_text_input("Chief complaint: ")

    return patient_data


def collect_operational_information():
    print("\n")
    print("############################################")
    print("#       OPERATIONAL INFORMATION")
    print("############################################")

    doctors = select_from_menu(
        "NUMBER OF AVAILABLE DOCTORS",
        ["1 doctor", "2 doctors", "3 doctors", "4 doctors"],
    )
    num_doctors = int(doctors.split()[0])

    peak = select_from_menu(
        "PEAK DURATION",
        ["120 minutes", "240 minutes", "360 minutes"],
    )
    peak_limit = int(peak.split()[0])

    print("\n============================================")
    print("FORECAST HORIZON")
    print("============================================")
    horizon = get_integer_input(
        "Hours to forecast (1-12): ", minimum=1, maximum=12
    )

    return num_doctors, peak_limit, horizon


def run_traditional_prediction(patient_data):
    pkg = load_triage_model()
    model = pkg["model"]
    pre = pkg.get("preprocessor", pkg["scaler"])
    feat_cols = pkg["feature_columns"]

    missing = [c for c in feat_cols if c not in patient_data]
    if missing:
        raise ValueError(
            "Missing triage features: " + ", ".join(missing)
        )

    df = pd.DataFrame([{c: patient_data[c] for c in feat_cols}])
    X = pre.transform(df)
    label = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(X)[0], dtype=float)
        classes = list(getattr(model, "classes_", [1, 2, 3, 4, 5]))
        aligned = np.zeros(5, dtype=float)
        for cls, prob in zip(classes, raw):
            idx = int(cls) - 1
            if 0 <= idx < 5:
                aligned[idx] = prob
        total = aligned.sum()
        if total > 0:
            aligned = aligned / total
        probs = aligned
    else:
        probs = np.zeros(5, dtype=float)
        if 1 <= label <= 5:
            probs[label - 1] = 1.0

    return label, probs


def run_nlp_prediction(chief_complaint):
    label, probs = predict_nlp_proba(chief_complaint)
    return int(label), np.asarray(probs, dtype=float)


def run_deep_learning_prediction(patient_data):
    model, pre, _ = load_deep_learning_model()
    pred, probs = predict_deep_learning_ktas(model, pre, patient_data)
    label = int(pred[0])
    flat = np.asarray(probs, dtype=float).reshape(-1)
    if flat.shape == (5,):
        aligned = flat
    else:
        aligned = np.zeros(5, dtype=float)
        if 1 <= label <= 5:
            aligned[label - 1] = 1.0
    total = aligned.sum()
    if total > 0:
        aligned = aligned / total
    return label, aligned


def combine_weighted(trad_label, trad_probs,
                     nlp_label, nlp_probs,
                     dl_label, dl_probs):
    return logic.combine_weighted(
        trad_label, trad_probs,
        nlp_label, nlp_probs,
        dl_label, dl_probs,
    )


def main():
    print("\n")
    print("############################################")
    print("#          PREDICT-CARE AI")
    print("#       INTEGRATED PREDICTION TEST")
    print("############################################")

    options = load_ktas_options()
    patient_data = collect_patient_data(options)
    num_doctors, peak_limit, horizon = collect_operational_information()

    print("\nRunning Traditional Machine Learning...")
    trad_ktas, trad_probs = run_traditional_prediction(patient_data)

    print("Running Natural Language Processing...")
    nlp_ktas, nlp_probs = run_nlp_prediction(
        patient_data["Chief_complain"]
    )

    print("Running Deep Learning...")
    dl_ktas, dl_probs = run_deep_learning_prediction(patient_data)

    print("Combining model predictions (weighted soft vote)...")
    combined_ktas, combined_conf, breakdown = combine_weighted(
        trad_ktas, trad_probs, nlp_ktas, nlp_probs, dl_ktas, dl_probs
    )

    priority_code, priority_name = logic.map_ktas_to_priority(combined_ktas)

    print("Checking current patient-flow period...")
    is_peak, peak_status = logic.determine_current_peak_status()

    print("Selecting scenario from staffing and peak settings...")
    scenario = logic.scenario_from_inputs(num_doctors, peak_limit, is_peak)

    print("Predicting doctor waiting time...")
    predicted_wait = logic.run_patient_flow_prediction(
        priority_code, is_peak, num_doctors, peak_limit, scenario
    )

    print("Running time-series demand forecast...")
    ts = logic.run_time_series_forecast(scenario, horizon_hours=horizon)

    print("Running consistency check...")
    cons = logic.check_consistency(
        scenario,
        predicted_wait,
        ts["mean"] if ts.get("available") else None,
        num_doctors,
    )

    print("\n")
    print("############################################")
    print("#          PREDICT-CARE AI RESULTS")
    print("############################################")

    print("\nTraditional ML KTAS:       " + str(trad_ktas))
    print("NLP KTAS:                  " + str(nlp_ktas))
    print("Deep Learning KTAS:        " + str(dl_ktas))
    print("Combined KTAS:             " + str(combined_ktas))
    print("Confidence:                "
          + format(combined_conf * 100, ".1f") + "%")

    print("\nNLP Emergency Category:    "
          + ("Emergency" if nlp_ktas <= 3 else "Non-Emergency"))

    print("\nOperational Priority:      " + priority_name)
    print("Operational Priority Code: " + str(priority_code))
    print("Scenario Used:             " + scenario)
    print("Current Period:            " + peak_status)
    print("Available Doctors:         " + str(num_doctors))
    print("Peak Duration:             " + str(peak_limit) + " minutes")
    print("Predicted Doctor Wait:     "
          + format(predicted_wait, ".2f") + " minutes")

    print("\n============================================")
    print("DEMAND FORECAST (TIME-SERIES)")
    print("============================================")
    if ts.get("available"):
        print("Model:                     " + str(ts["model"]))
        print("Horizon:                   "
              + str(ts["horizon"]) + " hours")
        for i, val in enumerate(ts["forecast"], start=1):
            print("  Hour +" + str(i) + ": "
                  + format(val, ".0f") + " patients")
        print("Average:                   "
              + format(ts["mean"], ".0f") + " patients/hour")
    else:
        print("Forecast unavailable:      " + ts.get("message", ""))

    print("\n============================================")
    print("CONSISTENCY CHECK")
    print("============================================")
    if cons.get("suspicious"):
        for msg in cons["messages"]:
            print("[!] " + msg)
    else:
        print("No contradiction detected between forecast,")
        print("staffing and predicted wait.")

    print("\n============================================")
    print("END OF PREDICT-CARE TEST")
    print("============================================")


if __name__ == "__main__":
    main()