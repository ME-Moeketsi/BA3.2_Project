"""
Predict-Care AI
Unified CLI Menu

Here I am merging ktas_input_menu.py and waiting_time_input.py into
one entry point. Both flows support typed OR spoken input and call
pipeline.route().

The waiting-time flow no longer asks for ArrivalTime; it derives it
from the system clock, scaled to the same 0–240 minute rolling
window that Data/patient_flow_cleaned.csv uses.
"""

from datetime import datetime

import speech_recognition as sr

from pipeline import route
from waiting_time_nlp import (
    understand_waiting_time_request,
    get_clarification_message,
)


# ============================================================
# SHARED HELPERS
# ============================================================

def get_number(question, minimum=0, maximum=None):
    while True:
        try:
            value = int(input(question))
            if value < minimum:
                print(f"Please enter a number of at least {minimum}.")
                continue
            if maximum is not None and value > maximum:
                print(f"Please enter a number no greater than {maximum}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a whole number.")


def get_arrival_time_from_system_clock():
    """
    Here I am deriving ArrivalTime from the system clock.

    Data/patient_flow_cleaned.csv models a rolling 240-minute peak
    window, so ArrivalTime is minutes-into-the-window, not a clock
    time. I convert wall-clock time to minutes-since-midnight and
    then take it modulo 240 to land in the same 0–240 range the
    regressors were trained on.
    """
    now = datetime.now()
    minutes_since_midnight = now.hour * 60 + now.minute
    return minutes_since_midnight % 240


def get_speech_input(prompt="Press Enter when you are ready to speak...",
                     duration=10, language="en-ZA"):
    """Record from the microphone and return recognised text, or None."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=1)

    try:
        with microphone as source:
            input(prompt)
            print(f"Recording for {duration} seconds. Speak now!")
            audio = recognizer.record(source, duration=duration)

        print("Converting speech to text...")
        text = recognizer.recognize_google(audio, language=language)
        print(f"Recognized: {text}")
        return text

    except sr.UnknownValueError:
        print("Speech could not be understood.")
    except sr.RequestError as error:
        print("Speech recognition service error:", error)
    except (OSError, AttributeError) as error:
        print("Microphone error:", error)

    return None


def ask_text_or_speech(prompt_text, allow_speech=True):
    """Ask the user to type or speak a short answer and return it."""
    if allow_speech:
        print("\nInput method:")
        print("1. Type")
        print("2. Speak")
        choice = input("Choice (1-2): ").strip()
        if choice == "2":
            return get_speech_input()

    return input(prompt_text).strip()


# ============================================================
# KTAS FLOW
# ============================================================

def run_ktas_flow():
    complaint = ask_text_or_speech("Enter patient chief complaint: ")

    if not complaint:
        print("No complaint received.")
        return

    result = route(complaint)

    if result["type"] != "ktas":
        print("\nPrediction could not be completed:")
        print(result.get("message", "Unknown error."))
        return

    r = result["result"]
    print("\nPREDICTION RESULT")
    print("-----------------")
    print(f"Chief Complaint: {r['chief_complaint']}")
    print(f"Predicted KTAS Level: {r['predicted_ktas']}")
    print(f"Emergency Category: {r['emergency_category']}")
    print(
        "\nNote: This prediction is for decision-support only "
        "and does not replace professional clinical assessment."
    )


# ============================================================
# WAITING-TIME FLOW
# ============================================================

def run_waiting_time_flow():
    print("\nWAITING-TIME PREDICTION")
    print("-----------------------")
    print("ENTER PATIENT INFORMATION")
    print("\nPRIORITY LEVELS")
    print("1 - Low")
    print("2 - Medium")
    print("3 - High")

    priority = get_number("Priority level (1-3): ", 1, 3)

    # Here I am replacing the manual ArrivalTime prompt with the system
    # clock, scaled to the simulation's 0–240 minute rolling window.
    arrival_time = get_arrival_time_from_system_clock()
    print(f"Arrival time (from system clock): {arrival_time} minutes "
          f"into the simulated 240-minute window")

    num_doctors = get_number("Number of available doctors: ", 1)

    while True:
        peak = input("Is this a peak arrival period? (yes/no): ").lower().strip()
        if peak in ("yes", "y"):
            is_peak = 1
            break
        if peak in ("no", "n"):
            is_peak = 0
            break
        print("Please enter yes or no.")

    patient_info = {
        "Priority": priority,
        "ArrivalTime": arrival_time,
        "NumDoctors": num_doctors,
        "IsPeakArrival": is_peak,
    }

    print("\nWHAT WOULD YOU LIKE TO PREDICT?")
    print("Example: How long will I wait for a doctor?")
    print("Example: What is the total waiting time?")
    print("Example: Predict both waiting times.")

    request = ask_text_or_speech("\nYour question (type or speak): ")

    if not request:
        print("No request received.")
        return

    intent = understand_waiting_time_request(request)

    if intent == "clarify":
        print("\n" + get_clarification_message(intent))
        while True:
            choice = input("1 = doctor, 2 = total, 3 = both: ").strip()
            if choice == "1":
                request = "doctor"
                break
            if choice == "2":
                request = "total"
                break
            if choice == "3":
                request = "both"
                break
            print("Please enter 1, 2, or 3.")

    result = route(request, patient_info=patient_info)

    print("\nNLP RESULTS")
    print("-----------")

    if result["type"] == "error":
        print(f"Prediction could not be completed: {result['message']}")
        return

    if result.get("needs_clarification"):
        print(result.get("message", ""))
        return

    print(f"Recognized intent: {result.get('intent', 'unknown')}")
    print(result.get("message", ""))

    if "doctor_minutes" in result and "total_minutes" in result:
        print("\nNumeric estimates:")
        print(f"  Doctor wait   : {result['doctor_minutes']:.1f} minutes")
        print(f"  Total in hosp.: {result['total_minutes']:.1f} minutes")

    print("\nPatient information used:")
    for feature, value in patient_info.items():
        print(f"  {feature}: {value}")

    print("\nNote: This system is for decision-support only.")


# ============================================================
# MAIN MENU
# ============================================================

def main():
    print("\n================================")
    print("         PREDICT-CARE AI")
    print("   Unified NLP Triage Menu")
    print("================================")

    while True:
        print("\nChoose a service:")
        print("1. KTAS prediction (chief complaint)")
        print("2. Waiting-time prediction")
        print("3. Exit")
        choice = input("Choice (1-3): ").strip()

        if choice == "1":
            run_ktas_flow()
        elif choice == "2":
            run_waiting_time_flow()
        elif choice == "3":
            print("\nExiting Predict-Care AI.")
            break
        else:
            print("Invalid choice. Please select 1, 2 or 3.")


if __name__ == "__main__":
    main()