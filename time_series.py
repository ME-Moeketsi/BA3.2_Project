"""
Predict-Care AI
Time-Series Analysis

Purpose:
    Analyse patient demand over time and evaluate simple
    time-series forecasting models.

The model is evaluated by hiding the final part of each
scenario and asking the model to predict those observations.

This avoids treating the end of the simulation as normal
future patient demand.

NOTE: The comments will be deleted in the final version of the code. They are here for
debugging purposes.
"""

# Here I am importing the tools I need for my time-series analysis.
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Here I am importing the time-series models I will compare.
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt
)

# Here I am importing the error measurements I use to evaluate
# how close my predictions are to the actual patient counts.
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "Data"
MODEL_DIR = "Models"
RESULTS_DIR = "Results"

DATASET_FILE = os.path.join(
    DATA_DIR,
    "patient_flow_timeseries.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "time_series_model.pkl"
)

EVALUATION_FILE = os.path.join(
    RESULTS_DIR,
    "time_series_evaluation.csv"
)

PLOT_FILE = os.path.join(
    RESULTS_DIR,
    "time_series_evaluation.png"
)

# Here I choose patient count as my main time-series variable.
VALUE_COLUMN = "patient_count"

# Here I use 60% of each scenario for training and 30% for testing.
TRAIN_RATIO = 0.60
TEST_RATIO = 0.30


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    # Here I make sure my required folders exist.
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_time_series_data():

    # Here I check that my time-series dataset exists.
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    # Here I load the CSV file into a pandas DataFrame.
    data = pd.read_csv(DATASET_FILE)

    # Here I convert DateTime into a proper date/time value.
    data["DateTime"] = pd.to_datetime(
        data["DateTime"]
    )

    # Here I arrange the observations chronologically.
    data = data.sort_values(
        ["Scenario", "DateTime"]
    ).reset_index(drop=True)

    print("\n========== TIME-SERIES DATA ==========")

    print(f"Records: {len(data)}")
    print(f"Scenarios: {data['Scenario'].nunique()}")

    print(
        f"Date range: "
        f"{data['DateTime'].min()} "
        f"to "
        f"{data['DateTime'].max()}"
    )

    print("\nRecords per scenario:")

    print(
        data["Scenario"]
        .value_counts()
        .sort_index()
    )

    return data


# ============================================================
# BASIC ANALYSIS
# ============================================================

def analyse_data(data):

    # Here I calculate the average patient count for each scenario.
    average_patients = (
        data.groupby("Scenario")[VALUE_COLUMN]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAverage patient count:")

    print(
        average_patients
        .round(2)
    )

    # Here I also look at average waiting time because it gives
    # useful operational context for the patient-demand series.
    if "avg_total_wait" in data.columns:

        average_wait = (
            data.groupby("Scenario")["avg_total_wait"]
            .mean()
            .sort_values(ascending=False)
        )

        print("\nAverage waiting time:")

        print(
            average_wait
            .round(2)
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(train, test):

    # Here I train a Simple Exponential Smoothing model.
    ses_model = SimpleExpSmoothing(
        train,
        initialization_method="estimated"
    ).fit()

    # Here I predict the same number of observations that
    # exist in my hidden test set.
    ses_prediction = ses_model.forecast(
        len(test)
    )

    # Here I make sure patient counts cannot become negative.
    ses_prediction = np.maximum(
        ses_prediction,
        0
    )

    # Here I calculate the SES errors.
    ses_mae = mean_absolute_error(
        test,
        ses_prediction
    )

    ses_rmse = np.sqrt(
        mean_squared_error(
            test,
            ses_prediction
        )
    )


    # Here I train a Holt trend model.
    holt_model = Holt(
        train,
        initialization_method="estimated"
    ).fit()

    # Here I predict the hidden test period using Holt.
    holt_prediction = holt_model.forecast(
        len(test)
    )

    # Here I make sure Holt cannot produce negative patient counts.
    holt_prediction = np.maximum(
        holt_prediction,
        0
    )

    # Here I calculate the Holt errors.
    holt_mae = mean_absolute_error(
        test,
        holt_prediction
    )

    holt_rmse = np.sqrt(
        mean_squared_error(
            test,
            holt_prediction
        )
    )


    # Here I compare the two models using MAE.
    if ses_mae <= holt_mae:

        selected_model = "Simple Exponential Smoothing"
        selected_prediction = ses_prediction
        selected_mae = ses_mae
        selected_rmse = ses_rmse

    else:

        selected_model = "Holt Trend"
        selected_prediction = holt_prediction
        selected_mae = holt_mae
        selected_rmse = holt_rmse


    return {
        "ses_mae": ses_mae,
        "ses_rmse": ses_rmse,
        "holt_mae": holt_mae,
        "holt_rmse": holt_rmse,
        "selected_model": selected_model,
        "selected_prediction": selected_prediction,
        "selected_mae": selected_mae,
        "selected_rmse": selected_rmse
    }


# ============================================================
# TIME-SERIES ANALYSIS
# ============================================================

def run_time_series_analysis():

    # Here I make sure my folders are ready.
    create_directories()

    # Here I load my patient-flow time-series dataset.
    data = load_time_series_data()

    # Here I display basic information about the data.
    analyse_data(data)

    all_results = []
    saved_models = {}

    plot_data = []


    # Here I analyse each scenario independently.
    for scenario in data["Scenario"].unique():

        print("\n" + "=" * 60)
        print(f"SCENARIO: {scenario}")
        print("=" * 60)

        scenario_data = (
            data[data["Scenario"] == scenario]
            .sort_values("DateTime")
            .copy()
        )

        # Here I extract the patient-count time series.
        series = (
            scenario_data
            .set_index("DateTime")[VALUE_COLUMN]
            .astype(float)
        )

        # Here I calculate the training size.
        # Here I calculate how many observations I will use for training.
        train_size = int(
            len(series) * TRAIN_RATIO
        )
        
        # Here I calculate how many observations I will use for testing.
        test_size = int(
            len(series) * TEST_RATIO
        )
        
        # I make sure there are enough observations for both sections.
        if train_size < 2 or test_size < 1:
        
            print(
                "Not enough observations for train/test evaluation."
            )
            continue
        
        # Here I split the time series chronologically.
        # I do NOT randomly shuffle time-series observations.
        train = series.iloc[:train_size]
        
        # Here I use the next 30% as my unseen test data.
        test = series.iloc[
            train_size:train_size + test_size
        ]
        
        # The remaining observations represent the end of the
        # simulation, so I exclude them from model evaluation.
        excluded = series.iloc[
            train_size + test_size:
        ]
        
        print(f"Total observations: {len(series)}")
        print(f"Training observations: {len(train)}")
        print(f"Testing observations: {len(test)}")
        print(f"Excluded shutdown observations: {len(excluded)}")

        # Here I compare my two forecasting approaches.
        results = evaluate_models(
            train,
            test
        )

        print("\nModel comparison:")

        print(
            f"Simple Exponential Smoothing "
            f"MAE: {results['ses_mae']:.2f}, "
            f"RMSE: {results['ses_rmse']:.2f}"
        )

        print(
            f"Holt Trend "
            f"MAE: {results['holt_mae']:.2f}, "
            f"RMSE: {results['holt_rmse']:.2f}"
        )

        print(
            f"\nSelected model: "
            f"{results['selected_model']}"
        )

        print(
            f"Selected MAE: "
            f"{results['selected_mae']:.2f}"
        )

        print(
            f"Selected RMSE: "
            f"{results['selected_rmse']:.2f}"
        )


        # ====================================================
        # SAVE TEST PREDICTIONS
        # ====================================================

        selected_prediction = pd.Series(
            results["selected_prediction"],
            index=test.index
        )

        # Here I save the actual and predicted values so I can
        # inspect exactly how my model performed.
        for index in test.index:

            all_results.append({
                "Scenario": scenario,
                "DateTime": index,
                "Actual_Patient_Count": test.loc[index],
                "Predicted_Patient_Count": selected_prediction.loc[index],
                "Model": results["selected_model"],
                "MAE": results["selected_mae"],
                "RMSE": results["selected_rmse"]
            })


        # ====================================================
        # REFIT SELECTED MODEL ON ALL AVAILABLE DATA
        # ====================================================

        # After evaluation, I train the selected approach using
        # the complete scenario. This saved model can later be
        # used by the Predict-Care system.
        if results["selected_model"] == "Simple Exponential Smoothing":

            final_model = SimpleExpSmoothing(
                series,
                initialization_method="estimated"
            ).fit()

        else:

            final_model = Holt(
                series,
                initialization_method="estimated"
            ).fit()


        saved_models[scenario] = {
            "model": results["selected_model"],
            "fitted_model": final_model,
            "value_column": VALUE_COLUMN
        }


        # Here I store information needed for the evaluation graph.
        plot_data.append({
            "scenario": scenario,
            "series": series,
            "train": train,
            "test": test,
            "prediction": selected_prediction,
            "model": results["selected_model"]
        })


    # ========================================================
    # SAVE EVALUATION RESULTS
    # ========================================================

    evaluation_data = pd.DataFrame(
        all_results
    )

    evaluation_data.to_csv(
        EVALUATION_FILE,
        index=False
    )

    print(
        f"\nEvaluation results saved to: "
        f"{EVALUATION_FILE}"
    )


    # ========================================================
    # SAVE MODELS
    # ========================================================

    joblib.dump(
        saved_models,
        MODEL_FILE
    )

    print(
        f"Models saved to: "
        f"{MODEL_FILE}"
    )


    # ========================================================
    # CREATE EVALUATION GRAPH
    # ========================================================

    create_evaluation_plot(
        plot_data
    )

    return evaluation_data


# ============================================================
# VISUALISATION
# ============================================================

def create_evaluation_plot(plot_data):

    # Here I create one graph for each scenario so that the
    # actual and predicted test values are easy to see.
    number_of_plots = len(plot_data)

    rows = 3
    columns = 2

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 12),
        sharey=False
    )

    axes = axes.flatten()

    for position, item in enumerate(plot_data):

        ax = axes[position]

        scenario = item["scenario"]
        train = item["train"]
        test = item["test"]
        prediction = item["prediction"]
        model = item["model"]

        # Here I plot the training observations.
        ax.plot(
            train.index,
            train.values,
            marker="o",
            label="Training Data"
        )

        # Here I plot the actual unseen test observations.
        ax.plot(
            test.index,
            test.values,
            marker="o",
            label="Actual Test Data"
        )

        # Here I plot what my selected model predicted
        # for those unseen observations.
        ax.plot(
            prediction.index,
            prediction.values,
            marker="x",
            linestyle="--",
            label="Forecast"
        )

        ax.axvline(
            test.index[0],
            linestyle=":",
            label="Test Start"
        )

        ax.set_title(
            scenario
        )

        ax.set_xlabel(
            "Date and Time"
        )

        ax.set_ylabel(
            "Patient Count"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend(
            fontsize=8
        )


    # If there are unused subplot positions, I remove them.
    for position in range(
        len(plot_data),
        len(axes)
    ):

        fig.delaxes(
            axes[position]
        )


    fig.suptitle(
        "Predict-Care AI - Time-Series Evaluation",
        fontsize=16
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_FILE,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Evaluation graph saved to: "
        f"{PLOT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n=============================================="
    )

    print(
        "       PREDICT-CARE AI"
    )

    print(
        "       TIME-SERIES ANALYSIS"
    )

    print(
        "=============================================="
    )

    run_time_series_analysis()

    print(
        "\nTime-series analysis completed successfully."
    )