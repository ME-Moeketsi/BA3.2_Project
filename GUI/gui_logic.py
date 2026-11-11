"""
gui_logic.py

Model and prediction logic for the Predict-Care AI Flet triage GUI.

This module contains no UI code. It loads the project's models and data,
runs the individual prediction components, combines their results, and
formats the final output for GUI/GUI.py.
"""

import os
import re
import sys
import threading

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(GUI_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from healthcare_ai import load_triage_model, load_patient_flow_model
from NLP.pipeline import predict_ktas_proba as _nlp_predict_proba
from deep_learning import (
    load_deep_learning_model,
    predict_ktas as predict_deep_learning_ktas,
)

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    sr = None
    SR_AVAILABLE = False

MODEL_WEIGHTS = {"nlp": 0.50, "dl": 0.30, "trad": 0.20}

HISTORICAL_WAIT_BY_SCENARIO = {
    "Baseline_2Doc_240Peak": 182.65,
    "Peak_120Min": 150.27,
    "Peak_360Min": 202.51,
    "Staffing_1Doc": 858.74,
    "Staffing_3Doc": 2.81,
    "Staffing_4Doc": 0.71,
}


def _resolve_data_dir():
    for candidate in ("Data", "data"):
        path = os.path.join(PROJECT_DIR, candidate)
        if os.path.isdir(path):
            return path
    return os.path.join(PROJECT_DIR, "Data")


DATA_DIR = _resolve_data_dir()
TRIAGE_DATASET = os.path.join(DATA_DIR, "ktas_cleaned.csv")
PATIENT_FLOW_DATASET = os.path.join(DATA_DIR, "patient_flow_cleaned.csv")
TIME_SERIES_MODEL_PATH = os.path.join(PROJECT_DIR, "Models", "time_series_model.pkl")
_TS_CACHE = None

CATEGORY_LABELS = {
    "Injury": {"1": "1 = No", "2": "2 = Yes"},
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


def _normalise_columns(data):
    """Strip whitespace and a possible UTF-8 BOM from every CSV column."""
    data.columns = (
        data.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return data


def display_options_for(column_name, raw_values):
    mapping = CATEGORY_LABELS.get(column_name)
    if not mapping:
        return list(raw_values)
    return [mapping.get(str(value).strip(), str(value)) for value in raw_values]


def raw_value_from_display(column_name, display_text):
    mapping = CATEGORY_LABELS.get(column_name)
    text = str(display_text or "").strip()
    if not mapping:
        return text
    if "=" in text:
        return text.split("=", 1)[0].strip()
    return text


# ---------------------------------------------------------------------------
# Load dropdown options
# ---------------------------------------------------------------------------
def load_ktas_options():
    if not os.path.exists(TRIAGE_DATASET):
        raise FileNotFoundError(f"KTAS dataset not found: {TRIAGE_DATASET}")
    data = _normalise_columns(pd.read_csv(TRIAGE_DATASET))
    required = ["Arrival mode", "Injury", "Mental", "Pain"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"Missing categorical columns: {missing}. Available columns: "
            f"{data.columns.tolist()}"
        )
    return {
        column: sorted(data[column].dropna().astype(str).str.strip().unique())
        for column in required
    }


def safe_load_ktas_options():
    try:
        return load_ktas_options(), None
    except Exception as exc:
        fallback = {
            "Arrival mode": ["1", "2", "3", "4", "5", "6", "7"],
            "Injury": ["1", "2"],
            "Mental": ["1", "2", "3", "4"],
            "Pain": ["0", "1"],
        }
        return fallback, str(exc)


# ---------------------------------------------------------------------------
# Build patient record
# ---------------------------------------------------------------------------
def build_patient_data(values):
    required = [
        "Group", "Sex", "Age", "Patients number per hour", "Arrival mode",
        "Injury", "Mental", "Pain", "NRS_pain", "SBP", "DBP", "HR",
        "RR", "BT", "Saturation", "Chief_complain",
    ]
    missing = [key for key in required if key not in values or values[key] in (None, "")]
    if missing:
        raise ValueError("Please provide values for: " + ", ".join(missing))
    try:
        patient = {
            "Group": int(values["Group"]),
            "Sex": int(values["Sex"]),
            "Age": int(float(values["Age"])),
            "Patients number per hour": int(float(values["Patients number per hour"])),
            "Arrival mode": raw_value_from_display("Arrival mode", values["Arrival mode"]),
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"One or more numeric fields are invalid: {exc}") from exc
    return patient


def map_ktas_to_priority(ktas):
    if ktas in (1, 2):
        return 3, "HIGH"
    if ktas == 3:
        return 2, "MEDIUM"
    return 1, "LOW"


# ---------------------------------------------------------------------------
# Peak status and scenario selection
# ---------------------------------------------------------------------------
def determine_current_peak_status():
    """Infer peak status from available dataset columns after normalising names.

    The source file may have padded column names. If only Scenario or only
    IsPeakArrival exists, use the available signal. If neither exists, raise a
    diagnostic error rather than silently inventing a peak status.
    """
    if not os.path.exists(PATIENT_FLOW_DATASET):
        raise FileNotFoundError(f"Patient-flow dataset not found: {PATIENT_FLOW_DATASET}")

    data = _normalise_columns(pd.read_csv(PATIENT_FLOW_DATASET))
    if data.empty:
        raise ValueError(f"Patient-flow dataset is empty: {PATIENT_FLOW_DATASET}")

    # Accept common variations in the dataset's naming convention.
    aliases = {
        "Scenario": ("scenario", "ScenarioName", "Scenario Name"),
        "IsPeakArrival": ("is_peak_arrival", "PeakArrival", "Is Peak Arrival"),
    }
    for canonical, alternatives in aliases.items():
        if canonical not in data.columns:
            for alternative in alternatives:
                if alternative in data.columns:
                    data = data.rename(columns={alternative: canonical})
                    break

    scenario_signal = None
    if "Scenario" in data.columns:
        scenarios = data["Scenario"].dropna().astype(str).str.strip()
        if not scenarios.empty:
            scenario_signal = scenarios.value_counts().index[0].lower()

    indicator_signal = None
    if "IsPeakArrival" in data.columns:
        indicator = pd.to_numeric(data["IsPeakArrival"], errors="coerce")
        indicator = indicator[indicator.isin([0, 1])]
        if not indicator.empty:
            indicator_signal = float(indicator.mean()) >= 0.5

    if scenario_signal is None and indicator_signal is None:
        raise ValueError(
            "Cannot determine peak status: patient_flow_cleaned.csv must have "
            "a usable 'Scenario' and/or 'IsPeakArrival' column.\n"
            f"Available columns after whitespace cleanup: {data.columns.tolist()}\n"
            f"Dataset path: {PATIENT_FLOW_DATASET}"
        )

    scenario_peak = False
    if scenario_signal is not None:
        scenario_peak = any(word in scenario_signal for word in ("peak", "120", "360"))

    if scenario_peak and indicator_signal is not None:
        is_peak = scenario_peak or indicator_signal
    elif scenario_signal is not None and indicator_signal is not None:
        # Prefer the explicit indicator when the scenario label does not
        # contain an explicit peak marker.
        is_peak = scenario_peak or indicator_signal
    elif indicator_signal is not None:
        is_peak = indicator_signal
    else:
        is_peak = scenario_peak

    return int(is_peak), ("PEAK" if is_peak else "NON-PEAK")


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


# ---------------------------------------------------------------------------
# Individual model runners
# ---------------------------------------------------------------------------
def _get_preprocessor(pkg, model_name):
    """Select preprocessor/scaler without eagerly evaluating a missing key."""
    preprocessor = pkg.get("preprocessor")
    if preprocessor is None:
        preprocessor = pkg.get("scaler")
    if preprocessor is None:
        raise ValueError(
            f"{model_name} model package contains neither 'preprocessor' nor 'scaler'."
        )
    return preprocessor


def run_traditional_prediction(patient_data):
    pkg = load_triage_model()
    model = pkg["model"]
    pre = _get_preprocessor(pkg, "Traditional triage")
    feature_columns = pkg["feature_columns"]
    missing = [column for column in feature_columns if column not in patient_data]
    if missing:
        raise ValueError("Missing triage features: " + ", ".join(missing))
    frame = pd.DataFrame([{column: patient_data[column] for column in feature_columns}])
    transformed = pre.transform(frame)
    label = int(model.predict(transformed)[0])
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(transformed)[0], dtype=float)
        classes = list(getattr(model, "classes_", [1, 2, 3, 4, 5]))
        probabilities = np.zeros(5, dtype=float)
        for cls, probability in zip(classes, raw):
            index = int(cls) - 1
            if 0 <= index < 5:
                probabilities[index] = probability
        total = probabilities.sum()
        if total > 0:
            probabilities /= total
    else:
        probabilities = np.zeros(5, dtype=float)
        if 1 <= label <= 5:
            probabilities[label - 1] = 1.0
    return label, probabilities


def run_nlp_prediction(chief_complaint):
    label, probabilities = _nlp_predict_proba(chief_complaint)
    return int(label), np.asarray(probabilities, dtype=float)


def run_deep_learning_prediction(patient_data):
    model, preprocessor, _ = load_deep_learning_model()
    prediction, probabilities = predict_deep_learning_ktas(model, preprocessor, patient_data)
    label = int(np.asarray(prediction).reshape(-1)[0])
    flat = np.asarray(probabilities, dtype=float).reshape(-1)
    if flat.shape == (5,):
        aligned = flat
    else:
        aligned = np.zeros(5, dtype=float)
        if 1 <= label <= 5:
            aligned[label - 1] = 1.0
    total = aligned.sum()
    if total > 0:
        aligned /= total
    return label, aligned


def run_patient_flow_prediction(priority_level, is_peak, number_of_doctors, peak_limit, scenario):
    pkg = load_patient_flow_model()
    model = pkg["model"]
    preprocessor = _get_preprocessor(pkg, "Patient-flow")
    feature_columns = list(pkg["feature_columns"])
    record = {
        "Priority": int(priority_level),
        "IsPeakArrival": int(is_peak),
        "NumDoctors": int(number_of_doctors),
        "PeakLimit": int(peak_limit),
        "ArrivalTime": max(0, int(peak_limit) // 2),
        "Scenario": scenario,
    }
    missing = [column for column in feature_columns if column not in record]
    if missing:
        raise ValueError(
            "Patient-flow model expects features not supplied by the GUI: "
            + ", ".join(missing)
            + ". Check the saved model's feature_columns and this input record."
        )
    frame = pd.DataFrame([{column: record[column] for column in feature_columns}])
    prediction = float(model.predict(preprocessor.transform(frame))[0])
    if not np.isfinite(prediction):
        raise ValueError("Patient-flow model returned a non-finite waiting-time prediction.")
    return max(0.0, prediction)


# ---------------------------------------------------------------------------
# Combine predictions
# ---------------------------------------------------------------------------
def combine_weighted(trad_label, trad_probs, nlp_label, nlp_probs, dl_label, dl_probs, weights=None):
    weights = weights or MODEL_WEIGHTS
    blended = (
        weights["trad"] * np.asarray(trad_probs, dtype=float)
        + weights["nlp"] * np.asarray(nlp_probs, dtype=float)
        + weights["dl"] * np.asarray(dl_probs, dtype=float)
    )
    total = blended.sum()
    if total > 0:
        blended /= total
    final_ktas = int(np.argmax(blended) + 1)
    confidence = float(blended[final_ktas - 1])
    return final_ktas, confidence, {
        "strategy": "weighted", "trad": trad_label, "nlp": nlp_label,
        "dl": dl_label, "combined_ktas": final_ktas,
        "confidence": confidence, "combined_probs": blended.tolist(),
        "weights": dict(weights),
    }


def combine_most_urgent(trad_label, nlp_label, dl_label):
    final = min(trad_label, nlp_label, dl_label)
    return final, {
        "strategy": "urgent", "trad": trad_label, "nlp": nlp_label,
        "dl": dl_label, "combined_ktas": final,
    }


# ---------------------------------------------------------------------------
# Time-series forecast
# ---------------------------------------------------------------------------
def _load_time_series_bundle():
    global _TS_CACHE
    if _TS_CACHE is not None:
        return _TS_CACHE
    if not os.path.exists(TIME_SERIES_MODEL_PATH):
        raise FileNotFoundError(
            f"Time-series model not found: {TIME_SERIES_MODEL_PATH}. "
            "Train it with the project's time_series.py script first."
        )
    bundle = joblib.load(TIME_SERIES_MODEL_PATH)
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError("time_series_model.pkl does not contain a scenario bundle.")
    _TS_CACHE = bundle
    return bundle


def run_time_series_forecast(scenario, horizon_hours=3):
    try:
        bundle = _load_time_series_bundle()
    except Exception as exc:
        return {"scenario": scenario, "model": None, "horizon": horizon_hours,
                "forecast": [], "mean": None, "available": False, "message": str(exc)}
    if scenario not in bundle:
        return {"scenario": scenario, "model": None, "horizon": horizon_hours,
                "forecast": [], "mean": None, "available": False,
                "message": "Scenario not in trained bundle: " + scenario}
    entry = bundle[scenario]
    fitted = entry.get("fitted_model")
    if fitted is None:
        return {"scenario": scenario, "model": entry.get("model"), "horizon": horizon_hours,
                "forecast": [], "mean": None, "available": False,
                "message": "Fitted model missing for this scenario."}
    try:
        raw = fitted.forecast(int(horizon_hours))
        values = [max(0.0, float(value)) for value in np.asarray(raw, dtype=float).reshape(-1)]
    except Exception as exc:
        return {"scenario": scenario, "model": entry.get("model"), "horizon": horizon_hours,
                "forecast": [], "mean": None, "available": False,
                "message": "Forecast failed: " + str(exc)}
    return {"scenario": scenario, "model": entry.get("model"), "horizon": horizon_hours,
            "forecast": values, "mean": float(np.mean(values)) if values else None,
            "available": True, "message": ""}


def check_consistency(scenario, predicted_wait, forecast_mean, num_doctors):
    historical = HISTORICAL_WAIT_BY_SCENARIO.get(scenario)
    warnings_list = []
    if historical is not None:
        ratio = predicted_wait / historical if historical > 0 else None
        if predicted_wait < 1.0 and historical > 30.0:
            warnings_list.append(
                f"Predicted wait ({predicted_wait:.2f} min) is far below the "
                f"historical average for {scenario} ({historical:.2f} min)."
            )
        elif ratio is not None and ratio > 5.0:
            warnings_list.append(
                f"Predicted wait ({predicted_wait:.2f} min) is far above the "
                f"historical average for {scenario} ({historical:.2f} min)."
            )
    if forecast_mean is not None and num_doctors <= 1 and forecast_mean > 100:
        warnings_list.append(
            f"Forecast demand ({forecast_mean:.0f} patients/hour) is high but only 1 doctor is assigned."
        )
    return {"suspicious": bool(warnings_list), "historical_wait": historical, "messages": warnings_list}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_full_pipeline(patient_data, num_doctors, peak_limit, strategy="weighted", ts_horizon=3):
    results = {}
    trad_label, trad_probs = run_traditional_prediction(patient_data)
    results["trad_ktas"] = trad_label
    nlp_label, nlp_probs = run_nlp_prediction(patient_data["Chief_complain"])
    results["nlp_ktas"] = nlp_label
    results["nlp_category"] = "Emergency" if nlp_label <= 3 else "Non-Emergency"
    dl_label, dl_probs = run_deep_learning_prediction(patient_data)
    results["dl_ktas"] = dl_label

    if strategy == "urgent":
        final_ktas, breakdown = combine_most_urgent(trad_label, nlp_label, dl_label)
        confidence = None
    else:
        final_ktas, confidence, breakdown = combine_weighted(
            trad_label, trad_probs, nlp_label, nlp_probs, dl_label, dl_probs
        )
    results.update({"combined_ktas": final_ktas, "combined_confidence": confidence,
                    "combined_breakdown": breakdown, "strategy": strategy})
    priority_code, priority_name = map_ktas_to_priority(final_ktas)
    results["priority_code"] = priority_code
    results["priority_name"] = priority_name

    is_peak, peak_name = determine_current_peak_status()
    results["is_peak"] = is_peak
    results["peak_name"] = peak_name
    scenario = scenario_from_inputs(int(num_doctors), int(peak_limit), is_peak)
    results["scenario"] = scenario
    wait = run_patient_flow_prediction(priority_code, is_peak, num_doctors, peak_limit, scenario)
    results["predicted_wait"] = wait
    ts = run_time_series_forecast(scenario, horizon_hours=ts_horizon)
    results["time_series"] = ts
    results["consistency"] = check_consistency(
        scenario, wait, ts["mean"] if ts.get("available") else None, int(num_doctors)
    )
    results["num_doctors"] = int(num_doctors)
    results["peak_limit"] = int(peak_limit)
    return results


# ---------------------------------------------------------------------------
# Format results for the GUI output field
# ---------------------------------------------------------------------------
def format_results(results):
    lines = ["=" * 40, "PREDICT-CARE AI RESULTS", "=" * 40, "",
             f"Traditional ML KTAS:   {results['trad_ktas']}",
             f"NLP KTAS:              {results['nlp_ktas']}",
             f"Deep Learning KTAS:    {results['dl_ktas']}", "",
             "=" * 40, "COMBINED RESULTS", "=" * 40,
             f"Strategy:              {results['strategy']}",
             f"Final KTAS:            {results['combined_ktas']}"]
    if results.get("combined_confidence") is not None:
        lines.append(f"Confidence:            {results['combined_confidence'] * 100:.1f}%")
    breakdown = results.get("combined_breakdown", {})
    if "combined_probs" in breakdown:
        lines.append("Per-level probabilities:")
        for level, probability in enumerate(breakdown["combined_probs"], start=1):
            lines.append(f"  KTAS {level}: {probability * 100:5.1f}%")
    lines.extend(["", "=" * 40, "DOWNSTREAM", "=" * 40,
                  f"Operational Priority:  {results['priority_name']} (code {results['priority_code']})",
                  f"Scenario Used:         {results['scenario']}",
                  f"Current Period:        {results['peak_name']}",
                  f"Available Doctors:     {results['num_doctors']}",
                  f"Peak Duration:         {results['peak_limit']} minutes", "",
                  f"Predicted Doctor Wait: {results['predicted_wait']:.2f} minutes",
                  "", "=" * 40, "DEMAND FORECAST (TIME-SERIES)", "=" * 40])
    ts = results.get("time_series", {})
    if ts.get("available"):
        lines.extend([f"Model:                 {ts['model']}",
                      f"Horizon:               {ts['horizon']} hours", "Forecasted Patients:"])
        for hour, value in enumerate(ts["forecast"], start=1):
            lines.append(f"  Hour +{hour}: {value:.0f}")
        lines.append(f"Average:               {ts['mean']:.0f} patients/hour")
    else:
        lines.append("Forecast unavailable:  " + ts.get("message", "Unknown reason"))
    consistency = results.get("consistency", {})
    lines.extend(["", "=" * 40, "CONSISTENCY CHECK", "=" * 40])
    if consistency.get("suspicious"):
        lines.extend("[!] " + message for message in consistency.get("messages", []))
    else:
        lines.extend(["No contradiction detected between forecast,", "staffing and predicted wait."])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dictation engine used by GUI/GUI.py
# ---------------------------------------------------------------------------
_VOICE_STATE = {"listening": False, "stop_event": None, "thread": None}
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100,
}


def _extract_number(text):
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        value = float(match.group(1))
        return int(value) if value.is_integer() else value
    total = 0
    found = False
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _NUMBER_WORDS:
            found = True
            value = _NUMBER_WORDS[word]
            total = 100 if value == 100 else total + value
    return total if found else None


def _extract_number_after(text, anchor_words, minimum, maximum):
    lower = text.lower()
    for anchor in anchor_words:
        index = lower.find(anchor)
        if index < 0:
            continue
        number = _extract_number(lower[index + len(anchor):index + len(anchor) + 60])
        if number is not None and minimum <= number <= maximum:
            return number
    return None


def parse_dictation(transcript, available_options):
    updates = {}
    text = (transcript or "").strip()
    if not text:
        return updates
    lower = text.lower()
    if re.search(r"\b(male|man|boy)\b", lower) and not re.search(r"\bfemale\b", lower):
        updates["Sex"] = "1"
    elif re.search(r"\b(female|woman|girl)\b", lower):
        updates["Sex"] = "2"
    age = _extract_number_after(lower, ["age", "years old", "year old", "aged"], 0, 120)
    if age is not None:
        updates["Age"] = str(int(age))
    sbp = _extract_number_after(lower, ["systolic", "sbp", "blood pressure"], 40, 300)
    dbp = _extract_number_after(lower, ["diastolic", "dbp"], 20, 200)
    bp_match = re.search(r"blood pressure\s+(?:is\s+)?(\d{2,3})\s*(?:over|on|by|/)\s*(\d{2,3})", lower)
    if bp_match:
        sbp, dbp = int(bp_match.group(1)), int(bp_match.group(2))
    if sbp is not None: updates["SBP"] = str(int(sbp))
    if dbp is not None: updates["DBP"] = str(int(dbp))
    for key, anchors, low, high in [
        ("HR", ["heart rate", "pulse", "hr is", "hr of"], 20, 250),
        ("RR", ["respiratory rate", "resp rate", "breaths per minute", "rr is"], 4, 80),
        ("BT", ["body temperature", "temperature", "temp"], 25, 45),
        ("Saturation", ["oxygen saturation", "oxygen sat", "saturation", "spo2"], 0, 100),
        ("NRS_pain", ["pain score", "pain level", "nrs", "pain is"], 0, 10),
    ]:
        value = _extract_number_after(lower, anchors, low, high)
        if value is not None:
            updates[key] = str(value if key == "BT" else int(value))
    if re.search(r"\b(no injury|not injured|no visible injury)\b", lower):
        updates["Injury"] = "1 = No"
    elif re.search(r"\b(has injury|injured|visible injury|there is injury)\b", lower):
        updates["Injury"] = "2 = Yes"
    if re.search(r"\b(not in pain|no pain|pain free)\b", lower):
        updates["Pain"] = "0"
    elif re.search(r"\b(in pain|has pain|is in pain)\b", lower):
        updates["Pain"] = "1"
    if re.search(r"\b(unresponsive|no response)\b", lower): updates["Mental"] = "4 = Unresponsive"
    elif re.search(r"\b(pain response|responds to pain)\b", lower): updates["Mental"] = "3 = Pain Response"
    elif re.search(r"\b(verbal response|responds verbally)\b", lower): updates["Mental"] = "2 = Verbal Response"
    elif re.search(r"\b(alert|oriented|awake)\b", lower): updates["Mental"] = "1 = Alert"
    if re.search(r"\b(private ambulance)\b", lower): updates["Arrival mode"] = "4 = Private Ambulance"
    elif re.search(r"\b(public ambulance|ambulance)\b", lower): updates["Arrival mode"] = "2 = Public Ambulance"
    elif re.search(r"\b(walking|walked in|ambulatory)\b", lower): updates["Arrival mode"] = "1 = Walking"
    elif re.search(r"\b(private vehicle|car|taxi)\b", lower): updates["Arrival mode"] = "3 = Private Vehicle"
    elif re.search(r"\bwheelchair\b", lower): updates["Arrival mode"] = "5 = Wheelchair"
    elif re.search(r"\b(stretcher|gurney)\b", lower): updates["Arrival mode"] = "6 = Stretcher"
    for marker in ["chief complaint", "complaining of", "complains of", "presenting with", "came in with", "the patient has"]:
        index = lower.find(marker)
        if index >= 0:
            complaint = text[index + len(marker):].strip(" .,")
            if complaint:
                updates["Chief_complain"] = complaint
            break
    else:
        if len(text) > 5:
            updates["Chief_complain"] = text.strip(" .,")
    return updates


def speech_available():
    if not SR_AVAILABLE:
        return False
    try:
        sr.Microphone.list_microphone_names()
        return True
    except Exception:
        return False


def _default_microphone():
    try:
        return sr.Microphone()
    except Exception:
        return None


def start_dictation(on_transcript, on_status, on_error):
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
        microphone = _default_microphone()
        if microphone is None:
            on_error("Microphone not available")
            _VOICE_STATE["listening"] = False
            return
        try:
            on_status("Preparing microphone...")
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                on_status("Listening. Speak now, press Space again to stop.")
                chunks = []
                while not stop_event.is_set():
                    try:
                        chunks.append(recognizer.listen(source, timeout=1, phrase_time_limit=8))
                    except sr.WaitTimeoutError:
                        continue
                if not chunks:
                    on_status("No speech captured.")
                    return
            on_status("Transcribing...")
            # The recognizer currently processes the first captured phrase.
            transcript = recognizer.recognize_google(chunks[0], language="en-US")
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
    if not _VOICE_STATE["listening"]:
        return False
    stop_event = _VOICE_STATE.get("stop_event")
    if stop_event is not None:
        stop_event.set()
    return True


def is_listening():
    return bool(_VOICE_STATE["listening"])
