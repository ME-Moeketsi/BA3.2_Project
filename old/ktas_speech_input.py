
import speech_recognition as sr
from old.predict_ktas import predict_ktas

# ----- INITIALIZE SPEECH RECOGNITION -----
recognizer = sr.Recognizer()

# ----- SELECT WORKING MICROPHONE -----
microphone = sr.Microphone(device_index=1)

# ----- RECORD PATIENT COMPLAINT -----
try:
    with microphone as source:
        print("\nPREDICT-CARE AI")
        print("Speech-Based Patient Triage Support")
        print("----------------------------------")
        print("Microphone ready.")

        input("\nPress Enter when you are ready to speak...")

        print("Recording for 10 seconds. Speak now!")
        audio = recognizer.record(source, duration=10)

    # ----- CONVERT SPEECH TO TEXT -----
    print("\nRecording finished.")
    print("Converting speech to text...")

    complaint = recognizer.recognize_google(
        audio,
        language="en-ZA"
    )

    print(f"\nRecognized Complaint: {complaint}")

    # ----- PREDICT KTAS USING NLP MODEL -----
    predicted_ktas, emergency_category = predict_ktas(
        complaint
    )

    # ----- DISPLAY PREDICTION -----
    print("\nPREDICTION RESULT")
    print("-----------------")
    print(f"Chief Complaint: {complaint}")
    print(f"Predicted KTAS Level: {predicted_ktas}")
    print(f"Emergency Category: {emergency_category}")

    print(
        "\nNote: This prediction is for decision-support only "
        "and does not replace professional clinical assessment."
    )

except sr.UnknownValueError:
    print("\nSpeech was recorded, but could not be understood.")

except sr.RequestError as error:
    print("\nSpeech recognition service error:", error)

except OSError as error:
    print("\nMicrophone error:", error)

except KeyboardInterrupt:
    print("\nRecording cancelled.")
