"""
Predict-Care AI
Unified Prediction Pipeline

Here I am consolidating the old predict_ktas.py and nlp_pipeline.py
into a single module. It loads the saved KTAS classifier once, exposes
predict_ktas(), provides the trained waiting-time predictors, and
routes natural-language requests to the right engine.
"""

from pathlib import Path

import joblib

from nlp_preprocessing import clean_text
from waiting_time_nlp import (
    understand_waiting_time_request,
    generate_waiting_time_response,
    get_clarification_message,
)


# ============================================================
# MODEL PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
KTAS_MODEL_PATH = PROJECT_DIR / "Models" / "ktas_nlp_model.pkl"
KTAS_TFIDF_PATH = PROJECT_DIR / "Models" / "ktas_tfidf_vectorizer.pkl"
WAIT_MODEL_PATH = PROJECT_DIR / "Models" / "waiting_time_model.pkl"

_KTAS_MODEL = None
_KTAS_TFIDF = None
_WAIT_BUNDLE = None


# ============================================================
# KTAS ARTIFACTS
# ============================================================

def _load_ktas_artifacts():
    global _KTAS_MODEL, _KTAS_TFIDF
    if _KTAS_MODEL is not None and _KTAS_TFIDF is not None:
        return _KTAS_MODEL, _KTAS_TFIDF

    if not KTAS_MODEL_PATH.is_file():
        raise FileNotFoundError(f"NLP model not found: {KTAS_MODEL_PATH}")
    if not KTAS_TFIDF_PATH.is_file():
        raise FileNotFoundError(f"TF-IDF vectorizer not found: {KTAS_TFIDF_PATH}")

    model = joblib.load(KTAS_MODEL_PATH)
    tfidf = joblib.load(KTAS_TFIDF_PATH)

    if not callable(getattr(model, "predict", None)):
        raise TypeError("The saved NLP model does not provide a predict() method.")
    if not callable(getattr(tfidf, "transform", None)):
        raise TypeError("The saved TF-IDF vectorizer does not provide a transform() method.")

    _KTAS_MODEL = model
    _KTAS_TFIDF = tfidf
    return model, tfidf


# ============================================================
# KTAS PREDICTION
# ============================================================

def predict_ktas(chief_complaint):
    if chief_complaint is None:
        raise ValueError("Chief complaint cannot be empty.")

    chief_complaint = str(chief_complaint).strip()
    if not chief_complaint:
        raise ValueError("Chief complaint cannot be empty.")

    cleaned = clean_text(chief_complaint)
    if not cleaned:
        raise ValueError("Chief complaint contains no usable letters or numbers.")

    model, tfidf = _load_ktas_artifacts()
    features = tfidf.transform([cleaned])

    model_features = getattr(model, "n_features_in_", None)
    if model_features is not None and model_features != features.shape[1]:
        raise ValueError(
            "The saved NLP model and TF-IDF vectorizer are incompatible: "
            f"model expects {model_features}, vectorizer produced {features.shape[1]}."
        )

    level = int(model.predict(features)[0])
    if level not in {1, 2, 3, 4, 5}:
        raise ValueError(f"The NLP model returned an unexpected KTAS level: {level}")

    return {
        "chief_complaint": chief_complaint,
        "cleaned_complaint": cleaned,
        "predicted_ktas": level,
        "emergency_category": "Emergency" if level <= 3 else "Non-Emergency",
    }


# ============================================================
# WAITING-TIME ARTIFACTS
# ============================================================

def _load_waiting_time_bundle():
    global _WAIT_BUNDLE
    if _WAIT_BUNDLE is not None:
        return _WAIT_BUNDLE

    if not WAIT_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Waiting-time model not found: {WAIT_MODEL_PATH}. "
            "Run `python NLP/train_waiting_time_model.py` first."
        )

    bundle = joblib.load(WAIT_MODEL_PATH)
    if not isinstance(bundle, dict):
        raise TypeError("Waiting-time model file did not contain a dict bundle.")
    for key in ("doctor_model", "total_model", "feature_columns"):
        if key not in bundle:
            raise TypeError(f"Waiting-time model bundle is missing '{key}'.")

    _WAIT_BUNDLE = bundle
    return bundle


# ============================================================
# WAITING-TIME PREDICTION
# ============================================================

def predict_waiting_time(patient_info, prediction_type):
    """
    Here I am predicting waiting-time minutes from the four patient
    features. If the model file is missing, this raises rather than
    fabricating a number — the caller decides how to surface it.
    """
    import pandas as pd  # local import keeps the module import cheap

    bundle = _load_waiting_time_bundle()
    feature_columns = bundle["feature_columns"]

    # Here I am validating the keys before building the feature row.
    for key in feature_columns:
        if key not in patient_info:
            raise ValueError(f"patient_info is missing required key: {key}")

    try:
        row = {
            "Priority":       int(patient_info["Priority"]),
            "ArrivalTime":    int(patient_info["ArrivalTime"]),
            "NumDoctors":     int(patient_info["NumDoctors"]),
            "IsPeakArrival":  int(patient_info["IsPeakArrival"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("patient_info values must be integers.") from exc

    # Here I am building a DataFrame with the exact column names the
    # regressors were fitted on. This removes the sklearn warning and,
    # more importantly, guarantees the column order can never drift.
    features = pd.DataFrame([row], columns=feature_columns)

    doctor_minutes = float(bundle["doctor_model"].predict(features)[0])
    total_minutes = float(bundle["total_model"].predict(features)[0])

    # Here I am enforcing the physical constraint that total length of
    # stay cannot be shorter than time-to-doctor.
    if total_minutes < doctor_minutes:
        total_minutes = doctor_minutes

    # Here I am guarding against negative model outputs.
    doctor_minutes = max(doctor_minutes, 0.0)
    total_minutes = max(total_minutes, doctor_minutes)

    return {
        "doctor_minutes": doctor_minutes,
        "total_minutes": total_minutes,
        "message": generate_waiting_time_response(
            prediction_type,
            doctor_minutes=doctor_minutes,
            total_minutes=total_minutes,
        ),
    }


# ============================================================
# ROUTER
# ============================================================

def route(request, patient_info=None):
    if patient_info is None:
        try:
            return {"type": "ktas", "result": predict_ktas(request)}
        except (ValueError, FileNotFoundError, TypeError) as exc:
            return {"type": "error", "message": str(exc)}

    intent = understand_waiting_time_request(request)

    if intent in {"clarify", "unknown"}:
        return {
            "type": "waiting_time",
            "intent": intent,
            "needs_clarification": True,
            "message": get_clarification_message(intent),
        }

    try:
        prediction = predict_waiting_time(patient_info, intent)
    except (ValueError, FileNotFoundError, TypeError) as exc:
        return {"type": "error", "message": str(exc)}

    return {
        "type": "waiting_time",
        "intent": intent,
        "doctor_minutes": prediction["doctor_minutes"],
        "total_minutes": prediction["total_minutes"],
        "message": prediction["message"],
    }


# ============================================================
# STATUS + SELF-TEST
# ============================================================

def pipeline_status():
    if KTAS_MODEL_PATH.is_file() and KTAS_TFIDF_PATH.is_file() and WAIT_MODEL_PATH.is_file():
        return "READY"
    return "PARTIAL"


def run_pipeline():
    status = pipeline_status()
    print("\n############################################")
    print("# PREDICT-CARE AI — PREDICTION PIPELINE")
    print("############################################")
    print(f"\nPipeline status: {status}")
    print(f"KTAS model:        {KTAS_MODEL_PATH.relative_to(PROJECT_DIR)}")
    print(f"TF-IDF vectorizer: {KTAS_TFIDF_PATH.relative_to(PROJECT_DIR)}")
    print(f"Waiting-time model:{WAIT_MODEL_PATH.relative_to(PROJECT_DIR)}")

    if not KTAS_MODEL_PATH.is_file() or not KTAS_TFIDF_PATH.is_file():
        raise FileNotFoundError(
            "KTAS artifacts missing. Run `python NLP/train_ktas_model.py` first."
        )

    example = predict_ktas("Open Wound")
    print("\nKTAS self-test OK.")
    print(f"  Open Wound -> KTAS {example['predicted_ktas']} ({example['emergency_category']})")

    if WAIT_MODEL_PATH.is_file():
        demo_info = {
            "Priority": 2, "ArrivalTime": 120,
            "NumDoctors": 2, "IsPeakArrival": 1,
        }
        result = predict_waiting_time(demo_info, "both")
        print("\nWaiting-time self-test OK.")
        print(f"  doctor wait : {result['doctor_minutes']:.1f} min")
        print(f"  total time  : {result['total_minutes']:.1f} min")
        print(f"  {result['message']}")
    else:
        print("\nWaiting-time model missing. "
              "Run `python NLP/train_waiting_time_model.py` to enable it.")

    return status


if __name__ == "__main__":
    run_pipeline()