"""
gui_logic.py

Model and prediction logic for the AI Triage GUI.

This module contains NO Tkinter code. It is imported by GUI.py
and GUI_Pipeline.py so that the UI layer and the model layer
stay separated.
"""

import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(GUI_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ============================================================
# MODEL IMPORTS
# ============================================================

from healthcare_ai import (
    load_triage_model,
    load_patient_flow_model,
)

# New unified NLP pipeline (replaces old.nlp_pipeline)
from NLP.pipeline import predict_ktas_proba as _nlp_predict_proba

from deep_learning import (
    load_deep_learning_model,
    predict_ktas as predict_deep_learning_ktas,
)


# ============================================================
# COMBINATION WEIGHTS
# ============================================================

# Weights reflect model quality for this project. Adjust once
# validation numbers are available. Must sum to 1.0.
MODEL_WEIGHTS = dict(nlp=0.50, dl=0.30, trad=0.20)


# ============================================================
# DATASET PATHS
# ============================================================

def _resolve_data_dir():
    """Prefer an existing 'Data' / 'data' folder, else default to 'Data'."""
    default = os.path.join(PROJECT_DIR, "Data")
    for name in ("Data", "data"):
        path = os.path.join(PROJECT_DIR, name)
        if os.path.isdir(path):
            return path
    return default


DATA_DIR = _resolve_data_dir()
TRIAGE_DATASET = os.path.join(DATA_DIR, "ktas_cleaned.csv")
PATIENT_FLOW_DATASET = os.path.join(DATA_DIR, "patient_flow_cleaned.csv")


# ============================================================
# CATEGORICAL DISPLAY LABELS
# ============================================================
# Maps raw dataset values -> human-readable labels shown in the GUI.
# The value before "=" is what the model receives; the text after
# "=" is only for display.

def _numbered(*names):
    """Build {"1": "1 = name1", "2": "2 = name2", ...}."""
    return {str(i): f"{i} = {name}" for i, name in enumerate(names, start=1)}


CATEGORY_LABELS = {
    "Injury": _numbered("No", "Yes"),
    "Mental": _numbered(
        "Alert",
        "Verbal Response",
        "Pain Response",
        "Unresponsive",
    ),
    "Arrival mode": _numbered(
        "Walking",
        "Public Ambulance",
        "Private Vehicle",
        "Private Ambulance",
        "Wheelchair",
        "Stretcher",
        "Other / Unknown",
    ),
    # Pain has no numeric mapping in the dataset; leave as-is.
}


def display_options_for(column_name, raw_values):
    """
    Return a list of strings to show in a combobox for the given
    column. If a mapping exists, each raw value is replaced by its
    display label. Otherwise the raw values are returned unchanged.
    """
    mapping = CATEGORY_LABELS.get(column_name)
    if not mapping:
        return list(raw_values)

    stripped = (str(v).strip() for v in raw_values)
    return [mapping.get(key, key) for key in stripped]


def raw_value_from_display(column_name, display_text):
    """
    Reverse of display_options_for: take whatever the combobox is
    showing and return the raw dataset value the model expects.
    """
    text = str(display_text).strip()

    if not CATEGORY_LABELS.get(column_name):
        return text

    head, sep, _ = text.partition("=")
    return head.strip() if sep else text


# ============================================================
# OPTIONS LOADING
# ============================================================

_CATEGORICAL_COLUMNS = ["Arrival mode", "Injury", "Mental", "Pain"]


def load_ktas_options():
    """
    Load categorical options directly from the training
    dataset so the GUI never hardcodes dataset values.
    """
    data = pd.read_csv(TRIAGE_DATASET)

    absent = [col for col in _CATEGORICAL_COLUMNS if col not in data.columns]
    if absent:
        raise ValueError("Missing categorical columns: " + ", ".join(absent))

    return {
        col: sorted(data[col].dropna().astype(str).str.strip().unique())
        for col in _CATEGORICAL_COLUMNS
    }


def safe_load_ktas_options():
    """
    Same as load_ktas_options, but returns a safe fallback
    if the dataset cannot be loaded. GUI.py uses this so
    the window still opens even if data is missing.
    """
    try:
        return load_ktas_options(), None
    except Exception as exc:
        fallback = {col: ["Unknown"] for col in _CATEGORICAL_COLUMNS}
        return fallback, str(exc)


# ============================================================
# PATIENT RECORD BUILDER
# ============================================================

def build_patient_data(values):
    """
    Convert the GUI's raw string values into the typed dict
    that run_predict.py would have produced.

    Categorical comboboxes show display labels (e.g. "1 = No"),
    so we translate them back to the raw dataset value ("1")
    before feeding the model.
    """
    patient = {
        "Group": int(values["Group"]),
        "Sex": int(values["Sex"]),
        "Age": int(float(values["Age"])),
        "Patients number per hour": int(values["Patients number per hour"]),
    }

    for col in ("Arrival mode", "Injury", "Mental", "Pain"):
        patient[col] = raw_value_from_display(col, values[col])

    for col in ("NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation"):
        patient[col] = float(values[col])

    patient["Chief_complain"] = str(values["Chief_complain"]).strip()
    return patient


# ============================================================
# KTAS -> PRIORITY
# ============================================================

def map_ktas_to_priority(ktas):
    """
    KTAS 1,2 -> HIGH (code 3)
    KTAS 3   -> MEDIUM (code 2)
    KTAS 4,5 -> LOW (code 1)
    """
    if ktas in (1, 2):
        return 3, "HIGH"
    if ktas == 3:
        return 2, "MEDIUM"
    return 1, "LOW"


# ============================================================
# PEAK STATUS
# ============================================================

_OFF_PEAK = (0, "NON-PEAK")
_PEAK_KEYWORDS = ("peak", "120", "360")


def determine_current_peak_status():
    """
    Derive PEAK / NON-PEAK from the historical patient-flow
    dataset, exactly like run_predict.py.
    """
    if not os.path.exists(PATIENT_FLOW_DATASET):
        return _OFF_PEAK

    data = pd.read_csv(PATIENT_FLOW_DATASET).dropna(
        subset=["Scenario", "IsPeakArrival"]
    )
    if data.empty:
        return _OFF_PEAK

    top_scenario = data["Scenario"].astype(str).value_counts().idxmax().lower()
    scenario_is_peak = any(word in top_scenario for word in _PEAK_KEYWORDS)

    peak_share = float(
        pd.to_numeric(data["IsPeakArrival"], errors="coerce").fillna(0).mean()
    )

    if scenario_is_peak or peak_share >= 0.5:
        return 1, "PEAK"
    return _OFF_PEAK

#===================================================================
#INDIVIDUAL MODEL RUNNERS (with probabilities)
#===================================================================


def run_traditional_prediction(patient_data):
    """
    returns (ktas_Label,  prob_vector_of_length_5).
    prob_vector[i] = probability of KTAS i+1.
    """
    pkg = load triage model()
    model = pkg["model"]
    pre = pkg.get("preprocessor", pkg["scaler"])
    feat_cols = pkg["feature_columns"]
    
    missing = [c for c in feat_cols if c not in patient_data]
    if missing:
        raise ValueError("Missing triage features:" + " , ".join(missing))
    
    df = pd.DataFrame([{c: patient_data[c] for c in feat_cols}])
    X = pre.transform(df)
    
    label = int(model.predict(X)[0])
    
    if hasattr(model,"predict_proba")
       raw = np.asarray(model.predict_proba(X)[0], dtype=float)
       classes =list(getattr(model, "classes_", [1, 2, 3, 4, 5]))
       aligned = np.zeros(5, dtype=float)
       for cls, pro in zip(classes,raw):
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
                   def run_nlp_predictions(chief_complaint):
                       """
                       returns (ktas_label, prob_vector_of_length_5) via the new
                       NLP/pipeline.py (predict_ktas_proba).
                       """
                       label, probs_list = _nlp_predict_proba(chief_complaint)
                       return int(label), np.asarray(probs_list, dtype=float)
                   
                   def run_deep_learning_prediction(patient_data):
                       """
                       returns (ktas_label, prob_vector_of_length_5).
                       predict_ktas() in deep_learning.py already returns probabilities.
                       """
                       model, pre, _ = load_deep_learning_model()
                       pred, probs = predict_deep_learning_ktas(model, pre, patient_data)
                       label = int(pred[0])
                       
                       flat = np.asarray(probs[0], dtype=float).reshape(-1)
                       if flat.shape ==(5,):
                           aligned = flat
                       else:
                           aligned = np.zeros(5, dtype=float)
                           if 1 <= label <= 5:
                               aligned[label - 1] = 1.0
                               
                               total = aligned.sum()
                               if total > 0:
                                   aligned = aligned / total
                                   
                                return label, aligned
                            
def run_patient_flow_prediction(priority_level, is_peak, number_of_doctors, peak_limits):
    
    pkg = load_patient_flow_model()
    model = pkg["model"]
    pre = pkg.get("preprocessor", pkg.get("scaler"))
    fcols = list(pkg["feature_columns"])
    
    if is_peak  and peak_limit == 120:
        scenario = "Peak_120Min"
    elif is_peak and peak_limits ==360:
        scenario = "Peak_360Min"
    else:
        scenario = "Baseline_2Doc_240Peak"
        
    record = {
        "Priority": int(priority_level),
        "IsPeakArrival": int(is_peak),
        "NumDoctors": int(number_of_doctor),
        "PeakLimits": int(peak_limits)
        "ArrivalTime": max(0, int(peak_limit) // 2),
        "Scenario": scenario,
        
    }
    
    missing = [c for c in fcols if c not in record]
    if missing:
        raise ValueError(
            "Patient-flow input missing features:" + ", ".join(missing)
            
        )
        
        df = pd. DataFrame([{c: record[c] for c in fcols}])
        return float(model.predict(pre.transform(df))[0])
    
def combine_weighted(trad_label, trad_probs,
                    nlp_label, nlp_probs,
                    dl_label, dl_probs,
                    weights=None):
"""    
Weighted soft-vote KTAS combination.


Returns (final_ktas, confidence, breakdown_dict).
"""


if weights is None:
    weights = MODEL_WEIGHTS
    
blended = (
    weights["trad"]*np.asarray(trad_probs, dtype=float)
    + weights["nlp"] * np.asarray(nlp_probs, dtype=float)
    + weights["d1"] * np.asarray(d1_probs, dtype=float)
)

total = blended. sum()
if total > 0:
    blended = blended / total
    
    final_ktas = int(np.argmax(blende) + 1)
    confidence = float(blended[final_ktas - 1])
    
    breakdown = {
        "strategy": "weighted",
        "trad": trad_label,
        "nlp": nlp_label,
        "dl": dl_label,
        "combined_ktas": final_ktas,
        "confidence": confidence,
        "combined_probs": blended.tolist(),
        "weights": dict(weights),
    }
    
    return final_ktas, confidence, breakdown

def combine_most_urgent(trad_label, nlp_label, dl_label):
    """
    Final KTAS = min(labels). Clinically safest — errs toward overtriage.

    Returns (final_ktas, breakdown_dict).
    """
    final = min(trad_label, nlp_label, dl_label)
    breakdown = {
        "strategy": "urgent",
        "trad": trad_label,
        "nlp": nlp_label,
        "dl": dl_label,
        "combined_ktas": final,
    }
    return final, breakdown


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline(patient_data, num_doctors, peak_limit,
                      strategy="weighted"):
    """
    Run every model, combine them into a single KTAS, then
    derive priority and waiting time from that combined result.

    strategy: "weighted" (soft vote) or "urgent" (min KTAS).
    """
    results = {}

    # 1. Traditional ML
    trad_label, trad_probs = run_traditional_prediction(patient_data)
    results["trad_ktas"] = trad_label

    # 2. NLP
    nlp_label, nlp_probs = run_nlp_prediction(patient_data["Chief_complain"])
    results["nlp_ktas"] = nlp_label
    results["nlp_category"] = (
        "Emergency" if nlp_label <= 3 else "Non-Emergency"
    )

    # 3. Deep Learning
    dl_label, dl_probs = run_deep_learning_prediction(patient_data)
    results["dl_ktas"] = dl_label

    # 4. Combine
    if strategy == "urgent":
        final_ktas, breakdown = combine_most_urgent(
            trad_label, nlp_label, dl_label
        )
        confidence = None
    else:
        final_ktas, confidence, breakdown = combine_weighted(
            trad_label, trad_probs,
            nlp_label, nlp_probs,
            dl_label, dl_probs,
        )

    results["combined_ktas"] = final_ktas
    results["combined_confidence"] = confidence
    results["combined_breakdown"] = breakdown
    results["strategy"] = strategy
    
    # 5. Priority from combined KTAS
        priority_code, priority_name = map_ktas_to_priority(final_ktas)
        results["priority_code"] = priority_code
        results["priority_name"] = priority_name
    
        # 6. Peak detection
        is_peak, peak_name = determine_current_peak_status()
        results["is_peak"] = is_peak
        results["peak_name"] = peak_name
    
        # 7. Patient-flow wait time
        wait = run_patient_flow_prediction(
            priority_code, is_peak, num_doctors, peak_limit
        )
        results["predicted_wait"] = wait
    
        results["num_doctors"] = int(num_doctors)
        results["peak_limit"] = int(peak_limit)
    
        return results
    
# ============================================================
# TEXT FORMATTER
# ============================================================

def format_results(results):
    """
    Turn the result dict into the text block that appears in
    the GUI's right-hand output panel.
    """
    
   breakdown: lines = []
    lines.append("=" * 40)
    lines.append("PREDICT-CARE AI RESULTS")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Traditional ML KTAS:   " + str(results["trad_ktas"]))
    lines.append("NLP KTAS:              " + str(results["nlp_ktas"]))
    lines.append("Deep Learning KTAS:    " + str(results["dl_ktas"]))
    lines.append("")

    lines.append("=" * 40)
    lines.append("COMBINED RESULTS")
    lines.append("=" * 40)
    lines.append("Strategy:              " + results["strategy"])
    lines.append("Final KTAS:            " + str(results["combined_ktas"]))
    if results.get("combined_confidence") is not None:
        lines.append(
            "Confidence:            "
            + format(results["combined_confidence"] * 100, ".1f")
            + "%"
        )

    breakdown = results.get("combined_breakdown", {})
    if "combined_probs" in breakdown:
        lines.append("Per-level probabilities:")
        for i, prob in enumerate(breakdown["combined_probs"], start=1):
            lines.append("  KTAS " + str(i) + ": "
                         + format(prob * 100, "5.1f") + "%")
    lines.append("")
    
     lines.append("=" * 40)
        lines.append("DOWNSTREAM")
        lines.append("=" * 40)
        lines.append(
            "Operational Priority:  "
            + results["priority_name"]
            + " (code "
            + str(results["priority_code"])
            + ")"
        )
        
        line.append("Current Period:   " + results["peak_name"])
        line.append("Available Doctors:   "str(results["num_doctors"]))
        lines.append(
            "Peak Duration:      "
            + str(results["peak_limit"])
            
            + "minutes"
        )
        
        lines.append("")
        lines.append(
            "Predicted Doctor Wait: "
            + format(results["prediction_wait"], ".2f")
            + "minutes"
        )
        return "\n" . join(lines)
    

    

    
    
    
    
                             
                    
                               
    
       
       
         
    
    

