"""
Predict-Care AI
Main Program

Here I am running the complete AI/ML pipeline.

The project contains:

1. Emergency Triage Machine Learning
   - Random Forest Classifier

2. Hospital Patient Flow Machine Learning
   - Random Forest Regressor

3. Emergency Triage Deep Learning
   - Neural Network Classifier

The Deep Learning model is kept separate from the
traditional Machine Learning models.
"""


# ============================================================
# MACHINE LEARNING IMPORTS
# ============================================================

from healthcare_ai import *


# ============================================================
# DEEP LEARNING IMPORTS
# ============================================================

from deep_learning import (
    load_triage_dataset as load_dl_triage_dataset,
    preprocess_triage_data as preprocess_dl_triage_data,
    prepare_deep_learning_data,
    train_deep_learning_model,
    evaluate_deep_learning_model,
    save_deep_learning_model,
    deep_learning_status
)
# ============================================================
# TIME-SERIES IMPORTS
# ============================================================

from time_series import run_time_series_analysis

# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # CREATE DIRECTORIES
    # ========================================================

    create_directories()

    print("\n")
    print("############################################")
    print("#       PREDICT-CARE AI SYSTEM")
    print("############################################")

    print("\nStarting AI system...")


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    system_status()


    # ========================================================
    # DATASET 1
    # EMERGENCY TRIAGE
    # RANDOM FOREST CLASSIFIER
    # ========================================================

    print("\n")
    print("############################################")
    print("# DATASET 1: EMERGENCY TRIAGE")
    print("# RANDOM FOREST CLASSIFIER")
    print("############################################")

    triage_data = load_triage_dataset()

    triage_data = preprocess_triage_data(
        triage_data
    )

    (
        triage_X_train,
        triage_X_test,
        triage_y_train,
        triage_y_test,
        triage_scaler,
        triage_features
    ) = prepare_triage_data(
        triage_data
    )

    triage_model = train_triage_model(
        triage_X_train,
        triage_y_train
    )

    triage_results = evaluate_triage_model(
        triage_model,
        triage_X_test,
        triage_y_test
    )

    save_triage_model(
        triage_model,
        triage_scaler,
        triage_features
    )


    # ========================================================
    # DATASET 2
    # HOSPITAL PATIENT FLOW
    # RANDOM FOREST REGRESSOR
    # ========================================================

    print("\n")
    print("############################################")
    print("# DATASET 2: HOSPITAL PATIENT FLOW")
    print("# RANDOM FOREST REGRESSOR")
    print("############################################")

    flow_data = load_patient_flow_dataset()

    flow_data = preprocess_patient_flow_data(
        flow_data
    )

    (
        flow_X_train,
        flow_X_test,
        flow_y_train,
        flow_y_test,
        flow_scaler,
        flow_features
    ) = prepare_patient_flow_data(
        flow_data
    )

    flow_model = train_patient_flow_model(
        flow_X_train,
        flow_y_train
    )

    flow_results = evaluate_patient_flow_model(
        flow_model,
        flow_X_test,
        flow_y_test
    )

    save_patient_flow_model(
        flow_model,
        flow_scaler,
        flow_features
    )


    # ========================================================
    # DEEP LEARNING
    # EMERGENCY TRIAGE NEURAL NETWORK
    # ========================================================

    print("\n")
    print("############################################")
    print("# DEEP LEARNING")
    print("# EMERGENCY TRIAGE NEURAL NETWORK")
    print("############################################")

    dl_data = load_dl_triage_dataset()

    dl_data = preprocess_dl_triage_data(
        dl_data
    )

    (
        dl_X_train,
        dl_X_test,
        dl_y_train,
        dl_y_test,
        dl_scaler,
        dl_features
    ) = prepare_deep_learning_data(
        dl_data
    )

    dl_model, dl_history = train_deep_learning_model(
        dl_X_train,
        dl_y_train,
        epochs=30,
        batch_size=32
    )

    dl_results = evaluate_deep_learning_model(
        dl_model,
        dl_X_test,
        dl_y_test
    )

    save_deep_learning_model(
        dl_model,
        dl_scaler,
        dl_features
    )
    # ========================================================
    # TIME-SERIES ANALYSIS
    # PATIENT DEMAND FORECASTING
    # I am calling the function, not copying the entire time-series program into the main file.

    print("\n")
    print("############################################")
    print("# TIME-SERIES ANALYSIS")
    print("# PATIENT DEMAND FORECASTING")
    print("############################################")

    time_series_results = run_time_series_analysis()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("############################################")
    print("#       PREDICT-CARE AI RESULTS")
    print("############################################")

    print(
        f"\nRandom Forest Triage Accuracy: "
        f"{triage_results['accuracy']:.4f}"
    )

    print(
        f"Deep Learning Triage Accuracy: "
        f"{dl_results['accuracy']:.4f}"
    )

    print(
        f"Patient Flow MAE: "
        f"{flow_results['mae']:.4f}"
    )

    print(
        f"Patient Flow RMSE: "
        f"{flow_results['rmse']:.4f}"
    )

    print(
        f"Patient Flow R²: "
        f"{flow_results['r2']:.4f}"
    )
    
    print(
        f"\nTime-Series Test MAE: "
        f"{time_series_results['MAE'].mean():.2f}"
    )

    print(
        f"Time-Series Test RMSE: "
        f"{time_series_results['RMSE'].mean():.2f}"
    )

    print("\n")
    print("============================================")
    print("MODEL FILES CREATED")
    print("============================================")

    print(
        "\nTraditional ML:"
    )

    print(
        "Models/triage_model.pkl"
    )

    print(
        "Models/patient_flow_model.pkl"
    )

    print(
        "\nDeep Learning:"
    )

    print(
        "Models/deep_learning_triage_model.keras"
    )

    print(
        "Models/deep_learning_triage_preprocessor.pkl"
    )
    print(
        "\nTime-Series:"
        )
    print(
            "Models/time_series_model.pkl"
        )
    print(
        "\nTime-Series Results:"
        )
    print(
        "Results/time_series_evaluation.csv"
        )
    print(
        "Results/time_series_evaluation.png"
        )
    print("\n")
    print("All models have been trained, evaluated and saved.")


# ============================================================
# POST-MODEL SECTION
# ============================================================
#
# IMPORTANT:
#
# The code below is NOT part of the initial training process.
#
# Use the trained models only AFTER:
#
# 1. Main.py has successfully completed.
# 2. The model files exist inside Models/.
# 3. You have checked the evaluation results.
#
# This section will later be used when we build the actual
# patient prediction part of Predict-Care AI.
#
# DO NOT add prediction code here yet.
#
# The next stage will be:
#
# Patient information
#        ↓
# Data preprocessing
#        ↓
# Trained model
#        ↓
# Prediction
#        ↓
# AI decision-support output
#
# ============================================================