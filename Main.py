from healthcare_ai import *

if __name__ == "__main__":

    # Here I am creating the directories that my project needs.
    create_directories()

    # Here I am displaying the current status of my AI system.
    system_status()

    # Here I am loading my emergency triage dataset.
    data = load_dataset()

    # Here I am exploring my dataset before preprocessing it.
    explore_dataset(data)

    # Here I am preprocessing my dataset.
    data = preprocess_data(data)

    print("\nPredict-Care AI is ready.")
    
    #Also not Complete i am going to make updates since i was testing them locally and separately