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

NOTE: The comments will be deleted in the final version of the code.
They are here for debugging purposes.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
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

# A scenario's shutdown tail is any run of observations whose value
# is below this fraction of the training mean. Those hours are
# excluded from the test window because they are not normal demand.
# Set to 0.0 to disable shutdown detection entirely.
SHUTDOWN_DROP_RATIO = 0.10


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_time_series_data():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    data = pd.read_csv(DATASET_FILE)

    data["DateTime"] = pd.to_datetime(
        data["DateTime"]
    )

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
    average_patients = (
        data.groupby("Scenario")[VALUE_COLUMN]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAverage patient count:")
    print(average_patients.round(2))

    if "avg_total_wait" in data.columns:
        average_wait = (
            data.groupby("Scenario")["avg_total_wait"]
            .mean()
            .sort_values(ascending=False)
        )
        print("\nAverage waiting time:")
        print(average_wait.round(2))


# ============================================================
# SPLIT HELPERS
# ============================================================

def find_shutdown_start(series, threshold_ratio):
    """
    Return the integer position of the first observation whose
    value falls below (threshold_ratio * mean of the first 60%).
    Returns len(series) if no such observation exists.

    This lets us trim the simulation's closing hours out of the
    test window so they don't distort the forecasting metric.
    """
    if threshold_ratio <= 0:
        return len(series)

    # Use only the earlier part of the series to compute the mean,
    # so the closing hours don't drag the average down.
    reference_end = max(2, int(len(series) * TRAIN_RATIO))
    reference_mean = float(
        series.iloc[:reference_end].mean()
    )
    if reference_mean <= 0:
        return len(series)

    threshold = threshold_ratio * reference_mean

    for position, value in enumerate(series.values):
        if value < threshold:
            return position

    return len(series)


def split_series(series, train_ratio, test_ratio, threshold_ratio):
    """
    Split a scenario series into (train, test, excluded, shutdown_start).

    - train  : first train_ratio of the series
    - test   : next test_ratio, capped by shutdown_start
    - excluded: everything after test
    """
    total = len(series)
    train_size = int(total * train_ratio)
    test_size = int(total * test_ratio)

    shutdown_start = find_shutdown_start(series, threshold_ratio)

    # Never allow the test slice to include shutdown hours.
    test_end = min(train_size + test_size, shutdown_start)

    train = series.iloc[:train_size]
    test = series.iloc[train_size:test_end]
    excluded = series.iloc[test_end:]

    return train, test, excluded, shutdown_start


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(train, test):
    ses_model = SimpleExpSmoothing(
        train,
        initialization_method="estimated"
    ).fit()

    ses_prediction = ses_model.forecast(len(test))
    ses_prediction = np.maximum(ses_prediction, 0)

    ses_mae = mean_absolute_error(test, ses_prediction)
    ses_rmse = np.sqrt(
        mean_squared_error(test, ses_prediction)
    )

    holt_model = Holt(
        train,
        initialization_method="estimated"
    ).fit()

    holt_prediction = holt_model.forecast(len(test))
    holt_prediction = np.maximum(holt_prediction, 0)

    holt_mae = mean_absolute_error(test, holt_prediction)
    holt_rmse = np.sqrt(
        mean_squared_error(test, holt_prediction)
    )

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
        "selected_rmse": selected_rmse,
    }


# ============================================================
# TIME-SERIES ANALYSIS
# ============================================================

def run_time_series_analysis():
    create_directories()

    data = load_time_series_data()
    analyse_data(data)

    all_results = []
    saved_models = {}
    plot_data = []
    summary_rows = []

    for scenario in data["Scenario"].unique():

        print("\n" + "=" * 60)
        print(f"SCENARIO: {scenario}")
        print("=" * 60)

        scenario_data = (
            data[data["Scenario"] == scenario]
            .sort_values("DateTime")
            .copy()
        )

        # --------------------------------------------------
        # Build the series and tag an explicit hourly frequency.
        # This removes the statsmodels "inferred frequency" warning.
        # --------------------------------------------------
        series = (
            scenario_data
            .set_index("DateTime")[VALUE_COLUMN]
            .astype(float)
        )
        series.index = pd.DatetimeIndex(series.index, freq="h")

        train, test, excluded, shutdown_start = split_series(
            series,
            TRAIN_RATIO,
            TEST_RATIO,
            SHUTDOWN_DROP_RATIO,
        )

        if len(train) < 2 or len(test) < 1:
            print(
                "Not enough observations for train/test evaluation "
                "after shutdown trimming. Skipping scenario."
            )
            continue

        print(f"Total observations: {len(series)}")
        print(f"Training observations: {len(train)}")
        print(f"Testing observations: {len(test)}")
        print(f"Excluded shutdown observations: {len(excluded)}")
        if shutdown_start < len(series):
            print(
                f"Shutdown detected at hour index {shutdown_start} "
                f"({series.index[shutdown_start]})"
            )

        results = evaluate_models(train, test)

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

        # --------------------------------------------------
        # Save test predictions
        # --------------------------------------------------
        selected_prediction = pd.Series(
            results["selected_prediction"],
            index=test.index,
        )

        for index in test.index:
            all_results.append({
                "Scenario": scenario,
                "DateTime": index,
                "Actual_Patient_Count": test.loc[index],
                "Predicted_Patient_Count": selected_prediction.loc[index],
                "Model": results["selected_model"],
                "MAE": results["selected_mae"],
                "RMSE": results["selected_rmse"],
            })

        # --------------------------------------------------
        # Refit selected model on all available data
        # --------------------------------------------------
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
            "value_column": VALUE_COLUMN,
        }

        plot_data.append({
            "scenario": scenario,
            "series": series,
            "train": train,
            "test": test,
            "prediction": selected_prediction,
            "model": results["selected_model"],
        })

        summary_rows.append({
            "Scenario": scenario,
            "Model": results["selected_model"],
            "MAE": results["selected_mae"],
            "RMSE": results["selected_rmse"],
        })

    # ========================================================
    # SAVE EVALUATION RESULTS
    # ========================================================
    evaluation_data = pd.DataFrame(all_results)

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
    joblib.dump(saved_models, MODEL_FILE)
    print(
        f"Models saved to: "
        f"{MODEL_FILE}"
    )

    # ========================================================
    # CREATE EVALUATION GRAPH
    # ========================================================
    create_evaluation_plot(plot_data)

    # ========================================================
    # SUMMARY TABLE
    # ========================================================
    print_summary_table(summary_rows)

    return evaluation_data


# ============================================================
# VISUALISATION
# ============================================================

def create_evaluation_plot(plot_data):

    if not plot_data:
        print("No scenarios to plot.")
        return

    number_of_plots = len(plot_data)

    rows = 3
    columns = 2

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 12),
        sharey=False,
    )

    axes = axes.flatten()

    for position, item in enumerate(plot_data):
        ax = axes[position]

        scenario = item["scenario"]
        train = item["train"]
        test = item["test"]
        prediction = item["prediction"]
        model = item["model"]

        ax.plot(
            train.index,
            train.values,
            marker="o",
            label="Training Data",
        )

        ax.plot(
            test.index,
            test.values,
            marker="o",
            label="Actual Test Data",
        )

        ax.plot(
            prediction.index,
            prediction.values,
            marker="x",
            linestyle="--",
            label="Forecast",
        )

        ax.axvline(
            test.index[0],
            linestyle=":",
            label="Test Start",
        )

        ax.set_title(f"{scenario} ({model})")
        ax.set_xlabel("Date and Time")
        ax.set_ylabel("Patient Count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for position in range(len(plot_data), len(axes)):
        fig.delaxes(axes[position])

    fig.suptitle(
        "Predict-Care AI - Time-Series Evaluation",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_FILE,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Evaluation graph saved to: {PLOT_FILE}")


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary_table(summary_rows):
    if not summary_rows:
        return

    print("\n" + "=" * 60)
    print("SCENARIO SUMMARY")
    print("=" * 60)

    header = (
        f"{'Scenario':<24} "
        f"{'Model':<32} "
        f"{'MAE':>8} "
        f"{'RMSE':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['Scenario']:<24} "
            f"{row['Model']:<32} "
            f"{row['MAE']:>8.2f} "
            f"{row['RMSE']:>8.2f}"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(
        "\n=============================================="
    )
    print("       PREDICT-CARE AI")
    print("       TIME-SERIES ANALYSIS")
    print(
        "=============================================="
    )

    run_time_series_analysis()

    print(
        "\nTime-series analysis completed successfully."
    )