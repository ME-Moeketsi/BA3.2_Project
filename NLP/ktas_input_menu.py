
import speech_recognition as sr
from predict_ktas import predict_ktas


# ----- SPEECH INPUT FUNCTION -----
def get_speech_input():
    recognizer = sr.Recognizer()

    # Working microphone on your laptop
    microphone = sr.Microphone(device_index=1)

    try:
        with microphone as source:
            print("\nMicrophone ready.")
            input("Press Enter when you are ready to speak...")

            print("Recording for 10 seconds. Speak now!")
            audio = recognizer.record(source, duration=10)

        print("\nConverting speech to text...")

        complaint = recognizer.recognize_google(
            audio,
            language="en-ZA"
        )

        print(f"\nRecognized Complaint: {complaint}")

        return complaint

    except sr.UnknownValueError:
        print("Speech could not be understood.")
        return None

    except sr.RequestError as error:
        print("Speech recognition service error:", error)
        return None

    except (OSError, AttributeError) as error:
        print("Microphone error:", error)
        return None


# ----- MAIN PROGRAM -----
def main():
    print("\n================================")
    print("         PREDICT-CARE AI")
    print("   NLP Patient Triage Support")
    print("================================")

    while True:
        print("\nChoose input method:")
        print("1. Enter complaint using text")
        print("2. Enter complaint using speech")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            complaint = input(
                "\nEnter patient chief complaint: "
            ).strip()

        elif choice == "2":
            complaint = get_speech_input()

        elif choice == "3":
            print("\nExiting Predict-Care AI.")
            break

        else:
            print("\nInvalid choice. Please select 1, 2 or 3.")
            continue

        if not complaint:
            print("\nNo complaint received. Please try again.")
            continue

        # ----- MAKE KTAS PREDICTION -----
        predicted_ktas, emergency_category = predict_ktas(
            complaint
        )

        # ----- DISPLAY RESULT -----
        print("\nPREDICTION RESULT")
        print("-----------------")
        print(f"Chief Complaint: {complaint}")
        print(f"Predicted KTAS Level: {predicted_ktas}")
        print(f"Emergency Category: {emergency_category}")

        print(
            "\nNote: This prediction is for decision-support only "
            "and does not replace professional clinical assessment."
        )


if __name__ == "__main__":
    main()
