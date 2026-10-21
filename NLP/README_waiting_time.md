# Predict-Care AI — Waiting-Time NLP Component

## 1. Overview

The Waiting-Time Natural Language Processing (NLP) component is part of the Predict-Care AI healthcare decision-support system.

Its purpose is to interpret natural-language questions about patient waiting times and generate understandable explanations of predictions supplied by the machine-learning component.

The Waiting-Time NLP component does not train machine-learning models or independently calculate waiting-time predictions.

## 2. Objectives

- Understand questions about doctor waiting time and total time in hospital.
- Identify the type of prediction requested by the user.
- Handle ambiguous questions through conversational clarification.
- Generate human-readable explanations from externally supplied prediction results.
- Support integration with the patient-flow prediction models and graphical user interface (GUI).

## 3. Supported Prediction Types

| Prediction type | Description | NLP intent |
|---|---|---|
| Doctor waiting time | Estimated time before a patient sees a doctor (`DocWaitTime`) | `doctor` |
| Total time in hospital | Estimated time spent waiting for and receiving the doctor's service (`TotalTimeInHospital`) | `total` |
| Both | Requests both estimates | `both` |

For the current project implementation:

**TotalTimeInHospital = DocWaitTime + DocServiceTime**

Medication waiting time (`MedWaitTime`) is excluded from the current prediction scope.

The total-time definition is specific to the current project and should not be interpreted as a complete clinical length-of-stay or discharge-time prediction.

## 4. Files

### `waiting_time_nlp.py`

Contains the core NLP functionality:

- `preprocess_text()` — normalizes user input.
- `understand_waiting_time_request()` — identifies the requested prediction type.
- `format_waiting_time()` — converts numerical minutes into readable durations.
- `generate_waiting_time_response()` — produces explanations from externally supplied estimates.
- `get_clarification_message()` — provides clarification messages for ambiguous or unrecognized requests.
- `main()` — provides an interactive terminal demonstration.

### `waiting_time_input.py`

Collects structured patient-flow information, including:

- Priority level (1 = Low, 2 = Medium, 3 = High)
- Arrival time in simulation minutes
- Number of available doctors
- Peak-arrival status

It also accepts a natural-language question and identifies the requested prediction.

The current terminal input component does not produce waiting-time estimates.

## 5. NLP Approach

The Waiting-Time NLP component uses **rule-based intent recognition**.

User questions are normalized and compared against predefined expressions associated with different prediction types.

The component currently recognizes five intents:

| Intent | Meaning |
|---|---|
| `doctor` | Doctor waiting-time request |
| `total` | Total time-in-hospital request |
| `both` | Request for both estimates |
| `clarify` | Ambiguous question requiring clarification |
| `unknown` | Unrecognized question |

Unlike the KTAS NLP component, the Waiting-Time NLP component does not currently use a trained text-classification model.

## 6. Example Interaction

**User:** What is my waiting time?

**NLP:** Would you like to know the waiting time before seeing a doctor, the total time in the hospital, or both?

**User:** Both.

**NLP:** You requested both doctor waiting time and total time-in-hospital predictions. A prediction from the waiting-time model is required to provide an estimate.

This demonstrates conversational clarification and intent recognition without generating artificial predictions.

## 7. Running the Component

From the project root directory, run:

```powershell
python NLP/waiting_time_nlp.py
```

To test the structured patient input component:

```powershell
python NLP/waiting_time_input.py
```

The current NLP functionality uses Python's standard library and does not require additional NLP packages.

## 8. Integration Guide

The machine-learning pipeline or GUI can import the NLP functions directly:

```python
from NLP.waiting_time_nlp import (
    understand_waiting_time_request,
    generate_waiting_time_response
)

intent = understand_waiting_time_request(
    "How long will I wait to see a doctor?"
)
```

Expected intent:

```text
doctor
```

Once the prediction model returns numerical estimates, they can be passed to the response generator:

```python
response = generate_waiting_time_response(
    prediction_type="both",
    doctor_minutes=25,
    total_minutes=40
)

print(response)
```

The values above are illustrative examples, not actual predictions.

The integration component is responsible for obtaining patient information, running the relevant prediction models, and supplying the results to the NLP response generator.

## 9. Current Functionality and Limitations

**Implemented:**
- Text preprocessing
- Rule-based intent recognition
- Conversational clarification in the terminal
- Natural-language response generation
- Duration formatting
- Validation of missing, negative and non-finite numerical estimates

**Current limitations:**
- Intent recognition relies on predefined phrases and may misinterpret unfamiliar wording.
- The NLP component does not predict waiting times independently.
- Integration with the waiting-time prediction models and GUI is pending.
- Patient-flow `Priority` uses three levels and should not be assumed equivalent to the five KTAS urgency levels.
- The current total-time definition excludes medication waiting time and other hospital processes.

## 10. Future Improvements

- Expand supported natural-language expressions.
- Create automated intent-recognition tests and evaluate classification performance.
- Improve conversational handling of ambiguous questions.
- Add speech recognition for spoken waiting-time requests.
- Test the NLP component within the integrated Predict-Care AI system.
- Refine explanations based on user feedback and integration testing.

## 11. Project Scope

This component forms part of the Predict-Care AI academic capstone project and is intended for healthcare decision-support demonstrations.

It is not clinically validated and should not be used as the sole basis for patient triage, treatment decisions or operational decisions in a real healthcare facility.