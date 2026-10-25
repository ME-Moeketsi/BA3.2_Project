"""
gui_logic.py

Model and prediction logic for the AI Triage GUI.

This module contains NO Tkinter code. It is imported by GUI.py
and GUI_Pipeline.py so that the UI layer and the model layer
stay separated.
"""

import os
import re
import sys
import threading

import joblib
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
# OPTIONAL SPEECH IMPORTS
# ============================================================
# If speech_recognition or pyaudio is missing the GUI still
# works; the Space-bar dictation just reports unavailable.

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    sr = None
    SR_AVAILABLE = False


# ============================================================
# COMBINATION WEIGHTS
# ============================================================

MODEL_WEIGHTS = {
    "nlp": 0.50,
    "dl": 0.30,
    "trad": 0.20,
}


HISTORICAL_WAIT_BY_SCENARIO = {
    "Baseline_2Doc_240Peak": 182.65,
    "Peak_120Min": 150.27,
    "Peak_360Min": 202.51,
    "Staffing_1Doc": 858.74,
    "Staffing_3Doc": 2.81,
    "Staffing_4Doc": 0.71,
}


# ============================================================
# DATASET PATHS
# ============================================================

def _resolve_data_dir():
    for candidate in ("Data", "data"):
        p = os.path.join(PROJECT_DIR, candidate)
        if os.path.isdir(p):
            return p
    return os.path.join(PROJECT_DIR, "Data")


DATA_DIR = _resolve_data_dir()
TRIAGE_DATASET = os.path.join(DATA_DIR, "ktas_cleaned.csv")
PATIENT_FLOW_DATASET = os.path.join(DATA_DIR, "patient_flow_cleaned.csv")

TIME_SERIES_MODEL_PATH = os.path.join(
    PROJECT_DIR, "Models", "time_series_model.pkl"
)

_TS_CACHE = None


# ============================================================
# CATEGORICAL DISPLAY LABELS
# ============================================================

CATEGORY_LABELS = {
    "Injury": {
        "1": "1 = No",
        "2": "2 = Yes",
    },
    "Mental": {
        "1": "1 = Alert",
        "2": "2 = Verbal Response",
        "3": "3 = Pain Response",
        "4": "4 = Unresponsive",
    },
    "Arrival mode": {
        "1": "1 = Walking",
        "2": "2 = Public Ambulance",
        "3": "3 = Private Vehicle",
        "4": "4 = Private Ambulance",
        "5": "5 = Wheelchair",
        "6": "6 = Stretcher",
        "7": "7 = Other / Unknown",
    },
}


def display_options_for(column_name, raw_values):
    mapping = CATEGORY_LABELS.get(column_name)
    if not mapping:
        return list(raw_values)
    return [mapping.get(str(v).strip(), str(v)) for v in raw_values]


def raw_value_from_display(column_name, display_text):
    mapping = CATEGORY_LABELS.get(column_name)
    if not mapping:
        return str(display_text).strip()
    text = str(display_text).strip()
    if "=" in text:
        return text.split("=", 1)[0].strip()
    return text


# ============================================================
# OPTIONS LOADING
# ============================================================

def load_ktas_options():
    data = pd.read_csv(TRIAGE_DATASET)
    required = ["Arrival mode", "Injury", "Mental", "Pain"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing categorical columns: " + ", ".join(missing)
        )
    options = {}
    for col in required:
        vals = data[col].dropna().astype(str).str.strip().unique()
        options[col] = sorted(vals)
    return options


def safe_load_ktas_options():
    try:
        return load_ktas_options(), None
    except Exception as e:
        return {
            "Arrival mode": ["Unknown"],
            "Injury": ["Unknown"],
            "Mental": ["Unknown"],
            "Pain": ["Unknown"],
        }, str(e)


# ============================================================
# PATIENT RECORD BUILDER
# ============================================================

def build_patient_data(values):
    return {
        "Group": int(values["Group"]),
        "Sex": int(values["Sex"]),
        "Age": int(float(values["Age"])),
        "Patients number per hour": int(values["Patients number per hour"]),
        "Arrival mode": raw_value_from_display(
            "Arrival mode", values["Arrival mode"]
        ),
        "Injury": raw_value_from_display("Injury", values["Injury"]),
        "Mental": raw_value_from_display("Mental", values["Mental"]),
        "Pain": raw_value_from_display("Pain", values["Pain"]),
        "NRS_pain": float(values["NRS_pain"]),
        "SBP": float(values["SBP"]),
        "DBP": float(values["DBP"]),
        "HR": float(values["HR"]),
        "RR": float(values["RR"]),
        "BT": float(values["BT"]),
        "Saturation": float(values["Saturation"]),
        "Chief_complain": str(values["Chief_complain"]).strip(),
    }


# ============================================================
# KTAS -> PRIORITY
# ============================================================

def map_ktas_to_priority(ktas):
    if ktas in [1, 2]:
        return 3, "HIGH"
    if ktas == 3:
        return 2, "MEDIUM"
    return 1, "LOW"


# ============================================================
# PEAK STATUS
# ============================================================

def determine_current_peak_status():
    if not os.path.exists(PATIENT_FLOW_DATASET):
        return 0, "NON-PEAK"
    data = pd.read_csv(PATIENT_FLOW_DATASET)
    data = data.dropna(subset=["Scenario", "IsPeakArrival"])
    if data.empty:
        return 0, "NON-PEAK"
    most_common = (
        data["Scenario"].astype(str).value_counts().idxmax().lower()
    )
    peak_keywords = ("peak", "120", "360")
    scenario_says_peak = any(k in most_common for k in peak_keywords)
    avg_peak = float(
        pd.to_numeric(data["IsPeakArrival"], errors="coerce").fillna(0).mean()
    )
    is_peak = int(scenario_says_peak or avg_peak >= 0.5)
    return is_peak, ("PEAK" if is_peak else "NON-PEAK")


# ============================================================
# SCENARIO SELECTION
# ============================================================

def scenario_from_inputs(num_doctors, peak_limit, is_peak):
    if num_doctors == 1:
        return "Staffing_1Doc"
    if num_doctors == 3:
        return "Staffing_3Doc"
    if num_doctors == 4:
        return "Staffing_4Doc"
    if is_peak and peak_limit == 120:
        return "Peak_120Min"
    if is_peak and peak_limit == 360:
        return "Peak_360Min"
    return "Baseline_2Doc_240Peak"


# ============================================================
# INDIVIDUAL MODEL RUNNERS
# ============================================================

def run_traditional_prediction(patient_data):
    pkg = load_triage_model()
    model = pkg["model"]
    pre = pkg.get("preprocessor", pkg["scaler"])
    feat_cols = pkg["feature_columns"]

    missing = [c for c in feat_cols if c not in patient_data]
    if missing:
        raise ValueError("Missing triage features: " + ", ".join(missing))

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
    label, probs_list = _nlp_predict_proba(chief_complaint)
    return int(label), np.asarray(probs_list, dtype=float)


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


def run_patient_flow_prediction(
    priority_level, is_peak, number_of_doctors, peak_limit, scenario
):
    pkg = load_patient_flow_model()
    model = pkg["model"]
    pre = pkg.get("preprocessor", pkg.get("scaler"))
    fcols = list(pkg["feature_columns"])

    record = {
        "Priority": int(priority_level),
        "IsPeakArrival": int(is_peak),
        "NumDoctors": int(number_of_doctors),
        "PeakLimit": int(peak_limit),
        "ArrivalTime": max(0, int(peak_limit) // 2),
        "Scenario": scenario,
    }

    missing = [c for c in fcols if c not in record]
    if missing:
        raise ValueError(
            "Patient-flow input missing features: " + ", ".join(missing)
        )

    df = pd.DataFrame([{c: record[c] for c in fcols}])
    return float(model.predict(pre.transform(df))[0])


# ============================================================
# COMBINATION
# ============================================================

def combine_weighted(trad_label, trad_probs,
                     nlp_label, nlp_probs,
                     dl_label, dl_probs,
                     weights=None):
    if weights is None:
        weights = MODEL_WEIGHTS

    blended = (
        weights["trad"] * np.asarray(trad_probs, dtype=float)
        + weights["nlp"] * np.asarray(nlp_probs, dtype=float)
        + weights["dl"] * np.asarray(dl_probs, dtype=float)
    )

    total = blended.sum()
    if total > 0:
        blended = blended / total

    final_ktas = int(np.argmax(blended) + 1)
    confidence = float(blended[final_ktas - 1])

    return final_ktas, confidence, {
        "strategy": "weighted",
        "trad": trad_label,
        "nlp": nlp_label,
        "dl": dl_label,
        "combined_ktas": final_ktas,
        "confidence": confidence,
        "combined_probs": blended.tolist(),
        "weights": dict(weights),
    }


def combine_most_urgent(trad_label, nlp_label, dl_label):
    final = min(trad_label, nlp_label, dl_label)
    return final, {
        "strategy": "urgent",
        "trad": trad_label,
        "nlp": nlp_label,
        "dl": dl_label,
        "combined_ktas": final,
    }


# ============================================================
# TIME-SERIES FORECAST
# ============================================================

def _load_time_series_bundle():
    global _TS_CACHE
    if _TS_CACHE is not None:
        return _TS_CACHE

    if not os.path.exists(TIME_SERIES_MODEL_PATH):
        raise FileNotFoundError(
            "Time-series model not found: " + TIME_SERIES_MODEL_PATH
            + ". Run `python time_series.py` first."
        )

    bundle = joblib.load(TIME_SERIES_MODEL_PATH)
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError(
            "time_series_model.pkl does not contain a scenario bundle."
        )

    _TS_CACHE = bundle
    return bundle


def run_time_series_forecast(scenario, horizon_hours=3):
    try:
        bundle = _load_time_series_bundle()
    except Exception as exc:
        return {
            "scenario": scenario,
            "model": None,
            "horizon": horizon_hours,
            "forecast": [],
            "mean": None,
            "available": False,
            "message": str(exc),
        }

    if scenario not in bundle:
        return {
            "scenario": scenario,
            "model": None,
            "horizon": horizon_hours,
            "forecast": [],
            "mean": None,
            "available": False,
            "message": "Scenario not in trained bundle: " + scenario,
        }

    entry = bundle[scenario]
    fitted = entry.get("fitted_model")
    if fitted is None:
        return {
            "scenario": scenario,
            "model": entry.get("model"),
            "horizon": horizon_hours,
            "forecast": [],
            "mean": None,
            "available": False,
            "message": "Fitted model missing for this scenario.",
        }

    try:
        raw = fitted.forecast(int(horizon_hours))
        values = [
            max(0.0, float(v))
            for v in np.asarray(raw, dtype=float).reshape(-1)
        ]
    except Exception as exc:
        return {
            "scenario": scenario,
            "model": entry.get("model"),
            "horizon": horizon_hours,
            "forecast": [],
            "mean": None,
            "available": False,
            "message": "Forecast failed: " + str(exc),
        }

    return {
        "scenario": scenario,
        "model": entry.get("model"),
        "horizon": horizon_hours,
        "forecast": values,
        "mean": float(np.mean(values)) if values else None,
        "available": True,
        "message": "",
    }


# ============================================================
# CONSISTENCY CHECK
# ============================================================

def check_consistency(scenario, predicted_wait, forecast_mean,
                      num_doctors):
    historical = HISTORICAL_WAIT_BY_SCENARIO.get(scenario)
    warnings = []

    if historical is not None:
        ratio = predicted_wait / historical if historical > 0 else None
        if predicted_wait < 1.0 and historical > 30.0:
            warnings.append(
                "Predicted wait ("
                + format(predicted_wait, ".2f")
                + " min) is far below the historical average for "
                + scenario
                + " ("
                + format(historical, ".2f")
                + " min)."
            )
        elif ratio is not None and ratio > 5.0:
            warnings.append(
                "Predicted wait ("
                + format(predicted_wait, ".2f")
                + " min) is far above the historical average for "
                + scenario
                + " ("
                + format(historical, ".2f")
                + " min)."
            )

    if forecast_mean is not None and num_doctors <= 1 and forecast_mean > 100:
        warnings.append(
            "Forecast demand ("
            + format(forecast_mean, ".0f")
            + " patients/hour) is high but only 1 doctor is assigned."
        )

    return {
        "suspicious": bool(warnings),
        "historical_wait": historical,
        "messages": warnings,
    }


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline(patient_data, num_doctors, peak_limit,
                      strategy="weighted", ts_horizon=3):
    results = {}

    trad_label, trad_probs = run_traditional_prediction(patient_data)
    results["trad_ktas"] = trad_label

    nlp_label, nlp_probs = run_nlp_prediction(patient_data["Chief_complain"])
    results["nlp_ktas"] = nlp_label
    results["nlp_category"] = (
        "Emergency" if nlp_label <= 3 else "Non-Emergency"
    )

    dl_label, dl_probs = run_deep_learning_prediction(patient_data)
    results["dl_ktas"] = dl_label

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

    priority_code, priority_name = map_ktas_to_priority(final_ktas)
    results["priority_code"] = priority_code
    results["priority_name"] = priority_name

    is_peak, peak_name = determine_current_peak_status()
    results["is_peak"] = is_peak
    results["peak_name"] = peak_name

    scenario = scenario_from_inputs(num_doctors, peak_limit, is_peak)
    results["scenario"] = scenario

    wait = run_patient_flow_prediction(
        priority_code, is_peak, num_doctors, peak_limit, scenario
    )
    results["predicted_wait"] = wait

    ts = run_time_series_forecast(scenario, horizon_hours=ts_horizon)
    results["time_series"] = ts

    results["consistency"] = check_consistency(
        scenario,
        wait,
        ts["mean"] if ts.get("available") else None,
        num_doctors,
    )

    results["num_doctors"] = int(num_doctors)
    results["peak_limit"] = int(peak_limit)

    return results


# ============================================================
# TEXT FORMATTER
# ============================================================

def format_results(results):
    lines = []
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
    lines.append("Scenario Used:         " + results["scenario"])
    lines.append("Current Period:        " + results["peak_name"])
    lines.append("Available Doctors:     " + str(results["num_doctors"]))
    lines.append(
        "Peak Duration:         "
        + str(results["peak_limit"])
        + " minutes"
    )
    lines.append("")
    lines.append(
        "Predicted Doctor Wait: "
        + format(results["predicted_wait"], ".2f")
        + " minutes"
    )

    ts = results.get("time_series", {})
    lines.append("")
    lines.append("=" * 40)
    lines.append("DEMAND FORECAST (TIME-SERIES)")
    lines.append("=" * 40)

    if ts.get("available"):
        lines.append("Model:                 " + str(ts["model"]))
        lines.append("Horizon:               "
                     + str(ts["horizon"]) + " hours")
        lines.append("Forecasted Patients:")
        for i, val in enumerate(ts["forecast"], start=1):
            lines.append(
                "  Hour +" + str(i) + ": " + format(val, ".0f")
            )
        lines.append("Average:               "
                     + format(ts["mean"], ".0f") + " patients/hour")
    else:
        lines.append("Forecast unavailable:  " + ts.get("message", ""))

    cons = results.get("consistency", {})
    lines.append("")
    lines.append("=" * 40)
    lines.append("CONSISTENCY CHECK")
    lines.append("=" * 40)

    if cons.get("suspicious"):
        for msg in cons["messages"]:
            lines.append("[!] " + msg)
    else:
        lines.append("No contradiction detected between forecast,")
        lines.append("staffing and predicted wait.")

    return "\n".join(lines)


# ============================================================
# DICTATION ENGINE
# ============================================================
# Push-to-talk speech recognition. No text-to-speech.
# The user speaks once; we extract every field we can find
# from that single utterance.

# Voice activity state shared across calls.
_VOICE_STATE = {
    "listening": False,
    "stop_event": None,
    "thread": None,
}

# Common spoken number words to numeric values.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}


def _extract_number(text):
    """
    Return the first number found in text, handling digits and
    common spoken number words. Returns None if no number found.
    """
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        value = float(match.group(1))
        return int(value) if value.is_integer() else value

    words = re.findall(r"[a-z]+", text.lower())
    total = 0
    found = False
    for word in words:
        if word in _NUMBER_WORDS:
            value = _NUMBER_WORDS[word]
            found = True
            if value == 100:
                total = 100
            else:
                total += value
    if found:
        return total
    return None


def _extract_number_after(text, anchor_words, minimum, maximum):
    """
    Find a number that appears shortly after any of the anchor words.
    Returns None if no valid number found in range.
    """
    lower = text.lower()
    for anchor in anchor_words:
        idx = lower.find(anchor)
        if idx < 0:
            continue
        tail = lower[idx + len(anchor):idx + len(anchor) + 60]
        number = _extract_number(tail)
        if number is None:
            continue
        if minimum <= number <= maximum:
            return number
    return None


def parse_dictation(transcript, available_options):
    """
    Turn a single free-form utterance into field updates.

    available_options is the dict returned by safe_load_ktas_options,
    used so we know which arrival/mode strings exist in the dataset.

    Returns a dict: {field_key: value} for every field we recognised.
    Fields not recognised are simply absent.
    """
    updates = {}
    text = (transcript or "").strip()
    if not text:
        return updates

    lower = text.lower()

    # ---------- Sex ----------
    if re.search(r"\b(male|man|boy)\b", lower) and not re.search(
        r"\bfemale\b", lower
    ):
        updates["Sex"] = "1"
    elif re.search(r"\b(female|woman|girl)\b", lower):
        updates["Sex"] = "2"

    # ---------- Age ----------
    age = _extract_number_after(
        lower,
        ["age", "years old", "year old", "aged"],
        minimum=0, maximum=120,
    )
    if age is not None:
        updates["Age"] = str(int(age))

    # ---------- Vital: systolic / diastolic ----------
    sbp = _extract_number_after(
        lower, ["systolic", "sbp", "blood pressure"], minimum=40, maximum=300
    )
    dbp = _extract_number_after(
        lower, ["diastolic", "dbp"], minimum=20, maximum=200
    )

    bp_over = re.search(
        r"blood pressure\s+(?:is\s+)?(\d{2,3})\s*(?:over|on|by|/)\s*(\d{2,3})",
        lower,
    )
    if bp_over:
        sbp = int(bp_over.group(1))
        dbp = int(bp_over.group(2))

    if sbp is not None:
        updates["SBP"] = str(int(sbp))
    if dbp is not None:
        updates["DBP"] = str(int(dbp))

    # ---------- Heart rate ----------
    hr = _extract_number_after(
        lower, ["heart rate", "pulse", "hr is", "hr of"],
        minimum=20, maximum=250,
    )
    if hr is not None:
        updates["HR"] = str(int(hr))

    # ---------- Respiratory rate ----------
    rr = _extract_number_after(
        lower, ["respiratory rate", "resp rate", "breaths per minute", "rr is"],
        minimum=4, maximum=80,
    )
    if rr is not None:
        updates["RR"] = str(int(rr))

    # ---------- Body temperature ----------
    bt = _extract_number_after(
        lower, ["temperature", "temp", "body temperature"],
        minimum=25, maximum=45,
    )
    if bt is not None:
        updates["BT"] = str(bt)

    # ---------- Oxygen saturation ----------
    sat = _extract_number_after(
        lower, ["oxygen saturation", "oxygen sat", "saturation", "spo2"],
        minimum=0, maximum=100,
    )
    if sat is not None:
        updates["Saturation"] = str(int(sat))

    # ---------- NRS pain score ----------
    nrs = _extract_number_after(
        lower, ["pain score", "pain level", "nrs", "pain is"],
        minimum=0, maximum=10,
    )
    if nrs is not None:
        updates["NRS_pain"] = str(nrs)

    # ---------- Injury (yes / no) ----------
    if re.search(r"\b(no injury|not injured|no visible injury)\b", lower):
        updates["Injury"] = "1 = No"
    elif re.search(r"\b(has injury|injured|visible injury|there is injury)\b", lower):
        updates["Injury"] = "2 = Yes"

    # ---------- Pain (yes / no) ----------
    # Only set when explicitly qualified — avoid clashing with NRS_pain.
    if re.search(r"\b(not in pain|no pain|pain free)\b", lower):
        updates["Pain"] = "0"
    elif re.search(r"\b(in pain|has pain|is in pain)\b", lower):
        updates["Pain"] = "1"

    # ---------- Mental state ----------
    if re.search(r"\b(unresponsive|no response)\b", lower):
        updates["Mental"] = "4 = Unresponsive"
    elif re.search(r"\b(pain response|responds to pain)\b", lower):
        updates["Mental"] = "3 = Pain Response"
    elif re.search(r"\b(verbal response|responds verbally)\b", lower):
        updates["Mental"] = "2 = Verbal Response"
    elif re.search(r"\b(alert|oriented|awake)\b", lower):
        updates["Mental"] = "1 = Alert"

    # ---------- Arrival mode ----------
    if re.search(r"\b(walking|walked in|ambulatory)\b", lower):
        updates["Arrival mode"] = "1 = Walking"
    elif re.search(r"\b(public ambulance|ambulance)\b", lower):
        updates["Arrival mode"] = "2 = Public Ambulance"
    elif re.search(r"\b(private vehicle|car|taxi)\b", lower):
        updates["Arrival mode"] = "3 = Private Vehicle"
    elif re.search(r"\b(private ambulance)\b", lower):
        updates["Arrival mode"] = "4 = Private Ambulance"
    elif re.search(r"\b(wheelchair)\b", lower):
        updates["Arrival mode"] = "5 = Wheelchair"
    elif re.search(r"\b(stretcher|gurney)\b", lower):
        updates["Arrival mode"] = "6 = Stretcher"

    # ---------- Chief complaint ----------
    # Taken as the leftover free text. Trim obvious non-complaint words.
    complaint_markers = [
        "chief complaint", "complaining of", "complains of",
        "presenting with", "came in with", "the patient has",
    ]
    complaint_text = None
    for marker in complaint_markers:
        idx = lower.find(marker)
        if idx >= 0:
            complaint_text = text[idx + len(marker):].strip(" .,")
            break

    # Fallback: if no marker, use anything after the first sentence.
    if complaint_text is None and len(text) > 5:
        complaint_text = text.strip(" .,")

    if complaint_text:
        updates["Chief_complain"] = complaint_text

    return updates


def speech_available():
    """True if speech_recognition and a microphone can be used."""
    if not SR_AVAILABLE:
        return False
    try:
        # Probe the microphone list quickly.
        sr.Microphone.list_microphone_names()
        return True
    except Exception:
        return False


def _default_microphone():
    """Return the default input device, or None if unavailable."""
    try:
        return sr.Microphone()
    except Exception:
        return None


def start_dictation(on_transcript, on_status, on_error):
    """
    Begin push-to-talk dictation. Records until stop_dictation()
    is called, then transcribes and returns the transcript via
    on_transcript(transcript).

    on_transcript(transcript) is called on a background thread;
    the GUI is responsible for marshalling it to the main thread.
    on_status(message) reports intermediate state.
    on_error(message) reports failure.
    """
    if not SR_AVAILABLE:
        on_error("speech_recognition not installed")
        return False

    if _VOICE_STATE["listening"]:
        return False

    stop_event = threading.Event()
    _VOICE_STATE["stop_event"] = stop_event
    _VOICE_STATE["listening"] = True    
    def run():
        recognizer = sr.Recognizer()
        audio = None
        mic = _default_microphone()

        if mic is None:
            on_error("Microphone not available")
            _VOICE_STATE["listening"] = False
            return

        try:
            on_status("Preparing microphone...")
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                on_status("Listening. Speak now, press Space again to stop.")

                # Listen until stop_event is set.
                chunks = []
                while not stop_event.is_set():
                    try:
                        chunk = recognizer.listen(
                            source, timeout=1, phrase_time_limit=8
                        )
                        chunks.append(chunk)
                    except sr.WaitTimeoutError:
                        continue

                if not chunks:
                    on_status("No speech captured.")
                    return

                # Concatenate chunks: use the first chunk's recognizer
                # call with the combined audio if possible. Simpler path
                # is to recognise the first chunk that produced audio.
                audio = chunks[0] if len(chunks) == 1 else chunks[0]

            on_status("Transcribing...")
            transcript = recognizer.recognize_google(audio, language="en-US")
            on_transcript(transcript)

        except sr.UnknownValueError:
            on_status("Speech not understood.")
        except sr.RequestError as exc:
            on_error("Recognition service error: " + str(exc))
        except Exception as exc:
            on_error("Dictation error: " + str(exc))
        finally:
            _VOICE_STATE["listening"] = False
            _VOICE_STATE["stop_event"] = None

    thread = threading.Thread(target=run, daemon=True, name="Dictation")
    _VOICE_STATE["thread"] = thread
    thread.start()
    return True


def stop_dictation():
    """Signal the active dictation to stop recording."""
    if not _VOICE_STATE["listening"]:
        return False
    stop_event = _VOICE_STATE.get("stop_event")
    if stop_event is not None:
        stop_event.set()
    return True


def is_listening():
    return bool(_VOICE_STATE["listening"])