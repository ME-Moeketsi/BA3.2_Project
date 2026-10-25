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
