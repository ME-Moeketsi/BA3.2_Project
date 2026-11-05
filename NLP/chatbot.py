"""
Predict-Care AI
Integrated Prediction Test (Chatbot Edition)

Same logic as the GUI's pipeline, but driven from a conversational interface.
Moved to the nlp/ folder.
"""

import os
import sys

import numpy as np
import pandas as pd

# Since we are now in nlp/chatbot.py, we need to go up two levels to find PROJECT_DIR
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from healthcare_ai import (
    load_triage_model,
    load_patient_flow_model,
)

# Changed from NLP.pipeline to pipeline since we are now inside NLP/
from pipeline import predict_ktas_proba as predict_nlp_proba

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


# ==================================================
# NEW: The function to be called by the GUI
# ==================================================
def run_integrated_test(patient_data, num_doctors, peak_limit, horizon):
    """Runs the full pipeline and returns a formatted string."""
    try:
        trad_ktas, trad_probs = run_traditional_prediction(patient_data)
        nlp_ktas, nlp_probs = run_nlp_prediction(patient_data["Chief_complain"])
        dl_ktas, dl_probs = run_deep_learning_prediction(patient_data)

        combined_ktas, combined_conf, breakdown = combine_weighted(
            trad_ktas, trad_probs, nlp_ktas, nlp_probs, dl_ktas, dl_probs
        )

        priority_code, priority_name = logic.map_ktas_to_priority(combined_ktas)
        is_peak, peak_status = logic.determine_current_peak_status()
        scenario = logic.scenario_from_inputs(num_doctors, peak_limit, is_peak)

        predicted_wait = logic.run_patient_flow_prediction(
            priority_code, is_peak, num_doctors, peak_limit, scenario
        )

        ts = logic.run_time_series_forecast(scenario, horizon_hours=horizon)

        cons = logic.check_consistency(
            scenario,
            predicted_wait,
            ts["mean"] if ts.get("available") else None,
            num_doctors,
        )

        # Build Output Text
        out = []
        out.append("### PREDICT-CARE AI RESULTS")
        out.append(f"Traditional ML KTAS:  {trad_ktas}")
        out.append(f"NLP KTAS:             {nlp_ktas}")
        out.append(f"Deep Learning KTAS:   {dl_ktas}")
        out.append(f"Combined KTAS:        {combined_ktas}")
        out.append(f"Confidence:           {combined_conf * 100:.1f}%")
        out.append("")
        out.append(f"NLP Emergency Category: {'Emergency' if nlp_ktas <= 3 else 'Non-Emergency'}")
        out.append("")
        out.append(f"Operational Priority: {priority_name} (Code: {priority_code})")
        out.append(f"Scenario Used:        {scenario}")
        out.append(f"Current Period:       {peak_status}")
        out.append(f"Available Doctors:    {num_doctors}")
        out.append(f"Peak Duration:        {peak_limit} minutes")
        out.append(f"Predicted Doctor Wait:{predicted_wait:.2f} minutes")
        out.append("")
        out.append("### DEMAND FORECAST (TIME-SERIES)")
        if ts.get("available"):
            out.append(f"Model:   {ts['model']}")
            out.append(f"Horizon: {ts['horizon']} hours")
            for i, val in enumerate(ts["forecast"], start=1):
                out.append(f"  Hour +{i}: {val:.0f} patients")
            out.append(f"Average: {ts['mean']:.0f} patients/hour")
        else:
            out.append(f"Forecast unavailable: {ts.get('message', '')}")

        out.append("")
        out.append("### CONSISTENCY CHECK")
        if cons.get("suspicious"):
            for msg in cons["messages"]:
                out.append(f"[!] {msg}")
        else:
            out.append("No contradiction detected between forecast,")
            out.append("staffing and predicted wait.")

        return "\n".join(out)

    except Exception as e:
        return f"Error during integrated prediction:\n{str(e)}"
