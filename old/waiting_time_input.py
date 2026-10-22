
# Predict-Care AI
# Waiting-Time NLP - Hybrid Input Component


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


def understand_request(request):
    text = request.lower().strip()

    if any(phrase in text for phrase in [
        "both",
        "all waiting times",
        "all predictions",
        "doctor and total",
        "total and doctor"
    ]):
        return "both"

    if any(phrase in text for phrase in [
        "total",
        "overall",
        "hospital time"
    ]):
        return "total"

    if any(phrase in text for phrase in [
        "doctor",
        "consultation",
        "physician"
    ]):
        return "doctor"

    if "wait" in text or "time" in text:
        return "clarify"

    return "unknown"


def main():
    print("\n===================================")
    print("PREDICT-CARE AI")
    print("Waiting-Time NLP Input")
    print("===================================")

    print("\nENTER PATIENT INFORMATION")

    print("\nPRIORITY LEVELS")
    print("1 - Low")
    print("2 - Medium")
    print("3 - High")

    priority = get_number("Priority level (1-3): ", 1, 3)

    arrival_time = get_number(
        "Arrival time (simulation minutes): "
    )

    num_doctors = get_number(
        "Number of available doctors: ", 1
    )

    while True:
        peak = input(
            "Is this a peak arrival period? (yes/no): "
        ).lower().strip()

        if peak in ["yes", "y"]:
            is_peak = 1
            break

        elif peak in ["no", "n"]:
            is_peak = 0
            break

        else:
            print("Please enter yes or no.")

    print("\nWHAT WOULD YOU LIKE TO PREDICT?")
    print("Example: How long will I wait for a doctor?")
    print("Example: What is the total waiting time?")
    print("Example: Predict both waiting times.")

    request = input("\nYour question: ")

    prediction_type = understand_request(request)

    if prediction_type == "clarify":
        print("\nWhich waiting time would you like to predict?")
        print("1 - Doctor waiting time")
        print("2 - Total waiting time")
        print("3 - Both waiting times")

        while True:
            choice = input("Enter your choice (1-3): ").strip()

            if choice == "1":
                prediction_type = "doctor"
                break

            elif choice == "2":
                prediction_type = "total"
                break

            elif choice == "3":
                prediction_type = "both"
                break

            else:
                print("Please enter 1, 2, or 3.")

    patient_information = {
        "Priority": priority,
        "ArrivalTime": arrival_time,
        "NumDoctors": num_doctors,
        "IsPeakArrival": is_peak,
    }

    print("\n===================================")
    print("NLP RESULTS")
    print("===================================")

    print(f"\nRecognized request: {prediction_type}")

    print("\nPatient information:")

    for feature, value in patient_information.items():
        print(f"{feature}: {value}")

    if prediction_type == "unknown":
        print("\nSorry, I could not understand your request.")
        print("Try asking about doctor or total waiting time.")

    else:
        print("\nPatient information collected successfully.")
        print("Waiting-time prediction model not connected yet.")

    print("\nNote: This system is for decision-support only.")


if __name__ == "__main__":
    main()
