"""Testable bridge between Predict-Care's GUI, NLP, and saved ML models.

No training or model downloads happen during prediction. A missing model is
reported explicitly. KTAS predictions are educational and cannot diagnose a
patient or determine real-life triage priority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from NLP.pipeline import predict_ktas, predict_waiting_time
from NLP.waiting_time_nlp import (
    get_clarification_message,
    understand_waiting_time_request,
)
from NLP.local_language_engine import LocalLanguageEngine, URGENT_RESPONSE


@dataclass(frozen=True)
class Assessment:
    complaint: str
    ktas: int
    urgency: str
    priority: int
    risk_notice: str
    doctor_minutes: Optional[float] = None
    total_minutes: Optional[float] = None
    waiting_intent: Optional[str] = None
    waiting_message: str = ""
    emergency_warning: str = ""


def operational_priority(ktas: int) -> tuple[int, str]:
    """Explicit project mapping, NOT a clinically validated prioritization."""
    if ktas not in (1, 2, 3, 4, 5):
        raise ValueError("KTAS level must be between 1 and 5.")
    if ktas <= 2:
        return 3, "Higher model-indicated urgency"
    if ktas == 3:
        return 2, "Intermediate model-indicated urgency"
    return 1, "Lower model-indicated urgency"


def validate_operations(arrival_time: int, num_doctors: int, is_peak: int) -> dict:
    """Validate within the feature domain observed in supplied simulation data."""
    try:
        arrival_time = int(arrival_time)
        num_doctors = int(num_doctors)
        is_peak = int(is_peak)
    except (ValueError, TypeError) as exc:
        raise ValueError("Arrival time, doctors, and peak status must be whole numbers.") from exc
    if not 3 <= arrival_time <= 1107:
        raise ValueError("Simulation arrival time must be between 3 and 1107 minutes.")
    if num_doctors not in (1, 2, 3, 4):
        raise ValueError("Doctors must be 1, 2, 3, or 4 (training-data range).")
    if is_peak not in (0, 1):
        raise ValueError("Peak arrival must be Yes or No.")
    return {
        "ArrivalTime": arrival_time,
        "NumDoctors": num_doctors,
        "IsPeakArrival": is_peak,
    }


def assess_patient(
    complaint: str,
    *,
    arrival_time: Optional[int] = None,
    num_doctors: Optional[int] = None,
    is_peak: Optional[int] = None,
    waiting_request: str = "predict both waiting times",
) -> Assessment:
    """Use the trained complaint NLP classifier, then the waiting-time models.

    The waiting-time priority derives from the NLP-estimated KTAS using the
    project's existing 1/2->3, 3->2, 4/5->1 mapping. No risk score/probability
    is invented and an urgent safety warning takes precedence over wait times.
    """
    if not isinstance(complaint, str) or not complaint.strip():
        raise ValueError("Enter the patient's chief complaint.")

    complaint = complaint.strip()
    is_urgent = LocalLanguageEngine.detect_urgent_symptoms(complaint)
    nlp = predict_ktas(complaint)
    ktas = nlp["predicted_ktas"]
    priority, urgency = operational_priority(ktas)
    risk_notice = (
        "Experimental complaint-text NLP only; age and vital signs are not "
        "used. Not a diagnosis or clinical triage assessment."
    )

    base = {
        "complaint": complaint,
        "ktas": ktas,
        "urgency": urgency,
        "priority": priority,
        "risk_notice": risk_notice,
    }
    if is_urgent:
        return Assessment(
            **base,
            emergency_warning=URGENT_RESPONSE,
            waiting_message="Waiting-time estimates withheld for urgent warning symptoms. "
                            "Seek emergency assessment rather than relying on a queue estimate.",
        )

    if arrival_time is None or num_doctors is None or is_peak is None:
        return Assessment(
            **base,
            waiting_message="Enter arrival-time simulation minutes, available doctors, and peak "
                            "status to calculate waiting times.",
        )

    operations = validate_operations(arrival_time, num_doctors, is_peak)
    request = str(waiting_request or "predict both waiting times").strip()
    intent = understand_waiting_time_request(request)
    if intent in ("unknown", "clarify"):
        return Assessment(
            **base,
            waiting_intent=intent,
            waiting_message=get_clarification_message(intent),
        )

    result = predict_waiting_time({**operations, "Priority": priority}, intent)
    return Assessment(
        **base,
        doctor_minutes=result["doctor_minutes"],
        total_minutes=result["total_minutes"],
        waiting_intent=intent,
        waiting_message=result["message"],
    )
