"""
Predict-Care AI
Main Program

Here I am coordinating the separate AI/ML components of my project.

The traditional machine-learning workflow trains:
1. Emergency triage using a Random Forest Classifier.
2. Hospital patient flow using a Random Forest Regressor.

The other components remain separate:
3. Emergency triage using a Deep Learning Neural Network.
4. Patient-demand forecasting using Time-Series Analysis.
5. Chief-complaint processing using Natural Language Processing.

Here I am running the traditional ML and Deep Learning scripts as separate
Python processes because each script manages its own current training and
preprocessing workflow.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import subprocess
import sys


# Here I am finding the project directory from this file's location.
PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_project_directories():
    """Here I am making sure the main project folders exist."""

    required_directories = [
        PROJECT_DIR / "Data",
        PROJECT_DIR / "Models",
        PROJECT_DIR / "Results",
        PROJECT_DIR / "NLP",
        PROJECT_DIR / "Tests",
        PROJECT_DIR / "Project Documentation",
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("Project directories are ready.")


# ============================================================
# RUN A STANDALONE PYTHON SCRIPT
# ============================================================

def run_python_script(script_path, component_name):
    """
    Here I am running a component using the same Python interpreter
    that is running Main.py.

    I am allowing the component's output to appear in the terminal.
    If the component fails, I stop the main workflow instead of
    reporting that the complete system succeeded.
    """

    script_path = PROJECT_DIR / script_path

    if not script_path.is_file():
        raise FileNotFoundError(
            f"{component_name} script was not found: {script_path}"
        )

    print("\n")
    print("=" * 60)
    print(f"STARTING: {component_name}")
    print("=" * 60)

    subprocess.run(
        [sys.executable, "-u", str(script_path)],
        cwd=PROJECT_DIR,
        check=True,
    )

    print(f"\n{component_name} completed successfully.")


# ============================================================
# RUN TIME-SERIES ANALYSIS
# ============================================================

def run_time_series_component():
    """Here I am calling the time-series workflow."""

    from time_series import run_time_series_analysis

    print("\n")
    print("=" * 60)
    print("STARTING: TIME-SERIES ANALYSIS")
    print("=" * 60)

    results = run_time_series_analysis()

    print("\nTime-series analysis completed successfully.")
    return results


# ============================================================
# RUN NATURAL LANGUAGE PROCESSING
# ============================================================

def run_nlp_component():
    """Here I am calling the NLP workflow."""

    from NLP.nlp_pipeline import run_nlp_pipeline

    print("\n")
    print("=" * 60)
    print("STARTING: NATURAL LANGUAGE PROCESSING")
    print("=" * 60)

    results = run_nlp_pipeline()

    print("\nNLP pipeline completed successfully.")
    return results


# ============================================================
# CHECK EXPECTED OUTPUT FILES
# ============================================================

def report_model_files():
    """Here I am checking which expected model files exist."""

    expected_files = [
        PROJECT_DIR / "Models" / "triage_model.pkl",
        PROJECT_DIR / "Models" / "triage_preprocessor.pkl",
        PROJECT_DIR / "Models" / "patient_flow_model.pkl",
        PROJECT_DIR / "Models" / "patient_flow_preprocessor.pkl",
        PROJECT_DIR / "Models" / "deep_learning_triage_model.keras",
        PROJECT_DIR / "Models" / "deep_learning_triage_preprocessor.pkl",
        PROJECT_DIR / "Models" / "time_series_model.pkl",
        PROJECT_DIR / "Results" / "time_series_evaluation.csv",
        PROJECT_DIR / "Results" / "time_series_evaluation.png",
    ]

    print("\n")
    print("=" * 60)
    print("PREDICT-CARE AI OUTPUT FILE CHECK")
    print("=" * 60)

    for file_path in expected_files:
        relative_path = file_path.relative_to(PROJECT_DIR)
        status = "FOUND" if file_path.is_file() else "NOT FOUND"
        print(f"{status}: {relative_path}")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    """Here I am coordinating the project's separate AI components."""

    print("\n")
    print("#" * 60)
    print("#             PREDICT-CARE AI SYSTEM")
    print("#" * 60)
    print("\nStarting the AI/ML workflow...")

    create_project_directories()

    # Here I am training and evaluating the traditional ML models.
    # healthcare_ai.py manages both the triage and patient-flow models.
    run_python_script(
        Path("healthcare_ai.py"),
        "TRADITIONAL MACHINE LEARNING",
    )

    # Here I am training and evaluating the separate neural network.
    run_python_script(
        Path("deep_learning.py"),
        "DEEP LEARNING",
    )

    # Here I am running patient-demand forecasting.
    run_time_series_component()

    # Here I am running the NLP pipeline for chief complaints.
    run_nlp_component()

    # Here I am checking for the expected saved model and result files.
    report_model_files()

    print("\n")
    print("=" * 60)
    print("PREDICT-CARE AI WORKFLOW COMPLETED")
    print("=" * 60)
    print(
        "All pipeline steps completed without a reported error. "
        "Review each component's evaluation output before interpreting "
        "its results."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
