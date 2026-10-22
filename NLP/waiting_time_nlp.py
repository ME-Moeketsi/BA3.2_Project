"""
Predict-Care AI
Waiting-Time NLP Component

Here I am keeping the intent parser and response formatter. The
interactive menu that used to live in this file has been moved into
main_menu.py, so this module is now a pure library imported by
pipeline.py and main_menu.py.
"""

import re
import math


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())

    return text


def understand_waiting_time_request(user_input):
    text = preprocess_text(user_input)

    if not text:
        return "unknown"

    both_phrases = [
        "both",
        "both predictions",
        "all predictions",
        "all waiting times",
        "both waiting times",
        "doctor and total",
        "total and doctor",
        "doctor and overall",
        "overall and doctor",
    ]

    if any(phrase in text for phrase in both_phrases):
        return "both"

    doctor_phrases = [
        "doctor",
        "physician",
        "consultation",
        "medical practitioner",
        "see a doctor",
        "see the doctor",
        "attend to me",
        "attend to the patient",
        "before treatment",
    ]

    total_phrases = [
        "total",
        "overall",
        "hospital time",
        "time in hospital",
        "time in the hospital",
        "spend in hospital",
        "spend in the hospital",
        "length of stay",
        "hospital stay",
        "leave the hospital",
        "discharged",
        "discharge time",
    ]

    doctor_requested = any(phrase in text for phrase in doctor_phrases)
    total_requested = any(phrase in text for phrase in total_phrases)

    if doctor_requested and total_requested:
        return "both"
    if doctor_requested:
        return "doctor"
    if total_requested:
        return "total"

    general_phrases = [
        "wait",
        "waiting",
        "how long",
        "how much time",
        "time estimate",
        "prediction",
    ]

    if any(phrase in text for phrase in general_phrases):
        return "clarify"

    return "unknown"


def format_waiting_time(minutes):
    if isinstance(minutes, bool):
        return None
    if not isinstance(minutes, (int, float)):
        return None
    if not math.isfinite(minutes) or minutes < 0:
        return None

    rounded_minutes = round(minutes)
    hours, remaining_minutes = divmod(rounded_minutes, 60)

    if hours == 0:
        unit = "minute" if remaining_minutes == 1 else "minutes"
        return f"{remaining_minutes} {unit}"

    hour_unit = "hour" if hours == 1 else "hours"

    if remaining_minutes == 0:
        return f"{hours} {hour_unit}"

    minute_unit = "minute" if remaining_minutes == 1 else "minutes"
    return f"{hours} {hour_unit} and {remaining_minutes} {minute_unit}"


def generate_waiting_time_response(
    prediction_type,
    doctor_minutes=None,
    total_minutes=None,
):
    doctor_time = format_waiting_time(doctor_minutes)
    total_time = format_waiting_time(total_minutes)

    if prediction_type == "doctor":
        if doctor_time is None:
            return (
                "A valid doctor waiting-time estimate "
                "is currently unavailable."
            )
        return (
            f"The patient is estimated to wait approximately "
            f"{doctor_time} before seeing a doctor."
        )

    if prediction_type == "total":
        if total_time is None:
            return (
                "A valid total time-in-hospital estimate "
                "is currently unavailable."
            )
        return (
            f"The estimated total time in the hospital is "
            f"approximately {total_time}. This includes the time "
            f"spent waiting for a doctor and the doctor's service time."
        )

    if prediction_type == "both":
        if doctor_time is None or total_time is None:
            return (
                "One or more waiting-time estimates "
                "are currently unavailable."
            )
        return (
            f"The patient is estimated to wait approximately "
            f"{doctor_time} before seeing a doctor. The estimated "
            f"total time in the hospital is approximately "
            f"{total_time}, including the doctor's service time."
        )

    return (
        "Please specify whether you want doctor waiting time, "
        "total time in hospital, or both predictions."
    )


def get_clarification_message(intent):
    if intent == "clarify":
        return (
            "Would you like to know the waiting time before seeing "
            "a doctor, the total time in the hospital, or both?"
        )
    if intent == "unknown":
        return (
            "Sorry, I could not understand your request. Please ask "
            "about doctor waiting time or total time in hospital."
        )
    return ""