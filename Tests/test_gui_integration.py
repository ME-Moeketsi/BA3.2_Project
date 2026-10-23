"""Model integration tests; can run headlessly with pytest."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from NLP.pipeline import predict_ktas, predict_waiting_time
from prediction_service import assess_patient, operational_priority, validate_operations


def test_ktas_model_is_loaded_and_predicts_valid_label():
    result = predict_ktas("headache")
    assert result["predicted_ktas"] in range(1, 6)
    assert result["chief_complaint"] == "headache"


def test_waiting_time_models_return_finite_nonnegative_times():
    result = predict_waiting_time(
        {"Priority": 2, "ArrivalTime": 120, "NumDoctors": 2, "IsPeakArrival": 1},
        "both"
    )
    assert 0 <= result["doctor_minutes"] <= result["total_minutes"]
    assert "minute" in result["message"] or "hour" in result["message"]


def test_complete_model_pipeline():
    response = assess_patient(
        "arm injury", arrival_time=120, num_doctors=2, is_peak=0,
        waiting_request="predict both waiting times"
    )
    assert 1 <= response.ktas <= 5
    assert 1 <= response.priority <= 3
    assert response.waiting_intent == "both"
    assert response.total_minutes >= response.doctor_minutes >= 0


def test_text_only_assessment_does_not_fake_waiting_time():
    result = assess_patient("headache")
    assert result.ktas in range(1, 6)
    assert result.doctor_minutes is None
    assert "Enter" in result.waiting_message


def test_intent_recognizes_doctor_and_total():
    doctor = assess_patient("arm injury", arrival_time=120, num_doctors=2, is_peak=1,
                            waiting_request="how long until I see a doctor")
    total = assess_patient("arm injury", arrival_time=120, num_doctors=2, is_peak=1,
                           waiting_request="what is the total hospital time")
    assert doctor.waiting_intent == "doctor"
    assert total.waiting_intent == "total"


def test_unknown_question_requests_clarification():
    response = assess_patient("arm injury", arrival_time=120, num_doctors=2, is_peak=1,
                              waiting_request="what is your favourite colour")
    assert response.doctor_minutes is None
    assert "Please" in response.waiting_message or "Sorry" in response.waiting_message


def test_urgent_symptoms_override_waiting_models():
    for complaint in ("chest pain and shortness of breath", "shortness of breath and chest pain",
                      "unconscious", "severe bleeding"):
        result = assess_patient(complaint, arrival_time=120, num_doctors=2, is_peak=1)
        assert result.emergency_warning
        assert result.doctor_minutes is None
        assert result.total_minutes is None


@pytest.mark.parametrize("parameters", [
    (100000, 2, 1), (120, 0, 1), (120, 5, 0), (120, 2, 3), ("clock", 2, 0)
])
def test_invalid_operational_inputs(parameters):
    with pytest.raises(ValueError):
        validate_operations(*parameters)


def test_priority_mapping():
    assert [operational_priority(x)[0] for x in range(1, 6)] == [3, 3, 2, 1, 1]
