"""
Predict-Care AI
Application Entry Point

Here I coordinate the already-trained AI components and run integration checks.
I do not retrain models when the application starts. Training remains in the
individual component scripts so normal startup is quick and predictable.

Run from the project root:
    python Main.py
"""

from pathlib import Path
import importlib.util
import sys
import traceback

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "Models"
DATA_DIRS = [PROJECT_DIR / "data", PROJECT_DIR / "Data"]


def print_heading(title):
    """Here I print consistent headings for the terminal interface."""
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def first_existing(*paths):
    """Here I return the first path that exists, or the first candidate."""
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def check_files():
    """Here I report actual file availability instead of assuming readiness."""
    print_heading("PREDICT-CARE AI — COMPONENT AND FILE STATUS")

    expected = {
        "Traditional ML: triage model": [MODEL_DIR / "triage_model.pkl"],
        "Traditional ML: patient-flow model": [MODEL_DIR / "patient_flow_model.pkl"],
        "Traditional ML: medication-wait model": [MODEL_DIR / "medication_wait_model.pkl"],
        "Traditional ML: combined-wait model": [MODEL_DIR / "combined_wait_model.pkl"],
        "Traditional ML: total-time model": [
            MODEL_DIR / "total_time_excluding_med_wait_model.pkl"
        ],
        "Deep Learning model": [MODEL_DIR / "deep_learning_triage_model.keras"],
        "Deep Learning preprocessor": [
            MODEL_DIR / "deep_learning_triage_preprocessor.pkl"
        ],
        "NLP classifier": [MODEL_DIR / "ktas_nlp_model.pkl"],
        "NLP TF-IDF vectorizer": [MODEL_DIR / "ktas_tfidf_vectorizer.pkl"],
        "Time-series model": [MODEL_DIR / "time_series_model.pkl"],
        "NLP pipeline script": [
            PROJECT_DIR / "NLP" / "nlp_pipeline.py",
            PROJECT_DIR / "nlp_pipeline.py",
        ],
        "Deep Learning script": [PROJECT_DIR / "deep_learning.py"],
        "Traditional ML engine": [PROJECT_DIR / "healthcare_ai.py"],
    }

    statuses = {}
    for label, candidates in expected.items():
        path = first_existing(*candidates)
        exists = path.is_file()
        statuses[label] = exists
        relative = path.relative_to(PROJECT_DIR) if path.is_relative_to(PROJECT_DIR) else path
        print(f"{'FOUND' if exists else 'MISSING':8} | {label:<39} | {relative}")

    print("\nNote: a file being present confirms availability, not model quality or clinical validity.")
    return statuses


def find_dataset(filename):
    """Here I look in the project's common data-folder spellings."""
    for directory in DATA_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def run_traditional_ml_checks():
    """Here I test saved-model loading, normal predictions, and input validation."""
    print_heading("TRADITIONAL ML — INTEGRATION AND EDGE-CASE CHECKS")

    try:
        import numpy as np
        import pandas as pd
        import healthcare_ai as ai
    except Exception as exc:
        print(f"FAIL | Could not import healthcare_ai: {exc}")
        return False

    passed = 0
    failed = 0

    def report(name, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"PASS | {name}" + (f" — {detail}" if detail else ""))
        else:
            failed += 1
            print(f"FAIL | {name}" + (f" — {detail}" if detail else ""))

    # Here I load each saved model package without starting a training run.
    loaders = [
        ("Load triage model", ai.load_triage_model),
        ("Load doctor-wait model", ai.load_patient_flow_model),
        ("Load medication-wait model", ai.load_medication_wait_model),
        ("Load combined-wait model", ai.load_combined_wait_model),
        ("Load total-time model", ai.load_total_time_excluding_med_wait_model),
    ]
    packages = {}
    for label, loader in loaders:
        try:
            packages[label] = loader()
            report(label, True)
        except Exception as exc:
            report(label, False, str(exc))

    # Here I get real, schema-matching records from the cleaned datasets.
    triage_path = find_dataset("ktas_cleaned.csv")
    flow_path = find_dataset("patient_flow_cleaned.csv")

    triage_record = None
    flow_record = None

    try:
        if triage_path is not None:
            triage_df = pd.read_csv(triage_path)
            triage_features = packages.get("Load triage model", {}).get(
                "feature_columns", getattr(ai, "TRIAGE_FEATURES", [])
            )
            if all(column in triage_df.columns for column in triage_features):
                triage_record = triage_df[triage_features].dropna().iloc[0].to_dict()
        report(
            "Find a valid triage example",
            triage_record is not None,
            str(triage_path) if triage_record is not None else "cleaned dataset/features unavailable",
        )
    except Exception as exc:
        report("Find a valid triage example", False, str(exc))

    try:
        if flow_path is not None:
            flow_df = pd.read_csv(flow_path)
            flow_features = packages.get("Load doctor-wait model", {}).get(
                "feature_columns", getattr(ai, "FLOW_FEATURES", [])
            )
            if all(column in flow_df.columns for column in flow_features):
                flow_record = flow_df[flow_features].dropna().iloc[0].to_dict()
        report(
            "Find a valid patient-flow example",
            flow_record is not None,
            str(flow_path) if flow_record is not None else "cleaned dataset/features unavailable",
        )
    except Exception as exc:
        report("Find a valid patient-flow example", False, str(exc))

    # Here I verify that the prediction functions work with the saved packages.
    if triage_record is not None and "Load triage model" in packages:
        try:
            result = ai.predict_triage(triage_record, packages["Load triage model"])
            valid = len(result) == 1 and int(result[0]) in {1, 2, 3, 4, 5}
            report("Triage prediction returns KTAS 1–5", valid, f"prediction={result}")
        except Exception as exc:
            report("Triage prediction returns KTAS 1–5", False, str(exc))
    else:
        report("Triage prediction returns KTAS 1–5", False, "No usable triage sample/model")

    flow_predictors = [
        ("Doctor-wait prediction", ai.predict_doctor_wait, "Load doctor-wait model"),
        ("Medication-wait prediction", ai.predict_medication_wait, "Load medication-wait model"),
        ("Combined-wait prediction", ai.predict_combined_wait, "Load combined-wait model"),
        (
            "Total-time prediction",
            ai.predict_total_time_excluding_med_wait,
            "Load total-time model",
        ),
    ]
    for label, predictor, package_key in flow_predictors:
        if flow_record is None or package_key not in packages:
            report(label, False, "No usable patient-flow sample/model")
            continue
        try:
            result = np.asarray(predictor(flow_record, packages[package_key])).reshape(-1)
            valid = result.size == 1 and np.isfinite(result[0]) and result[0] >= 0
            report(label, valid, f"prediction={np.round(result, 3).tolist()}")
        except Exception as exc:
            report(label, False, str(exc))

    # Here I check rejection of common invalid patient-flow input.
    if flow_record is not None and "Load doctor-wait model" in packages:
        invalid_cases = []

        missing = dict(flow_record)
        missing.pop("ArrivalTime", None)
        invalid_cases.append(("Missing required feature", missing, (ValueError,)))

        nan_value = dict(flow_record)
        nan_value["ArrivalTime"] = np.nan
        invalid_cases.append(("NaN arrival time", nan_value, (ValueError,)))

        infinite_value = dict(flow_record)
        infinite_value["ArrivalTime"] = np.inf
        invalid_cases.append(("Infinite arrival time", infinite_value, (ValueError,)))

        negative_value = dict(flow_record)
        negative_value["ArrivalTime"] = -1
        invalid_cases.append(("Negative arrival time", negative_value, (ValueError,)))

        bad_priority = dict(flow_record)
        bad_priority["Priority"] = 99
        invalid_cases.append(("Unsupported priority", bad_priority, (ValueError,)))

        bad_scenario = dict(flow_record)
        bad_scenario["Scenario"] = "__UNKNOWN_SCENARIO__"
        invalid_cases.append(("Unknown scenario", bad_scenario, (ValueError,)))

        for label, bad_input, expected_errors in invalid_cases:
            try:
                ai.predict_doctor_wait(bad_input, packages["Load doctor-wait model"])
                report(label, False, "invalid input was accepted")
            except expected_errors as exc:
                report(label, True, str(exc))
            except Exception as exc:
                report(label, False, f"unexpected exception: {type(exc).__name__}: {exc}")

        # Here I verify that an empty batch is rejected.
        try:
            ai.predict_doctor_wait(pd.DataFrame(), packages["Load doctor-wait model"])
            report("Empty batch", False, "empty input was accepted")
        except (ValueError, TypeError) as exc:
            report("Empty batch", True, str(exc))

    # Here I verify that invalid triage numeric values are rejected.
    if triage_record is not None and "Load triage model" in packages:
        bad_triage = dict(triage_record)
        bad_triage["Age"] = np.inf
        try:
            ai.predict_triage(bad_triage, packages["Load triage model"])
            report("Infinite triage age", False, "invalid input was accepted")
        except (ValueError, TypeError) as exc:
            report("Infinite triage age", True, str(exc))
        except Exception as exc:
            report("Infinite triage age", False, f"unexpected exception: {exc}")

        # Unknown text categories should not crash a fitted OneHotEncoder that
        # was configured with handle_unknown='ignore'. I test compatibility,
        # not clinical appropriateness of the resulting prediction.
        unknown_category = dict(triage_record)
        unknown_category["Chief_complain"] = "__UNSEEN_TEST_CATEGORY__"
        try:
            result = ai.predict_triage(unknown_category, packages["Load triage model"])
            valid = len(result) == 1 and int(result[0]) in {1, 2, 3, 4, 5}
            report("Unseen triage text category handled safely", valid, f"prediction={result}")
        except Exception as exc:
            report("Unseen triage text category handled safely", False, str(exc))

    print(f"\nTraditional ML checks: {passed} passed, {failed} failed.")
    return failed == 0



def run_nlp_checks():
    """Here I test the saved NLP model through its public prediction function."""
    print_heading("NLP — INTEGRATION CHECKS")
    passed = 0
    failed = 0

    def report(name, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"PASS | {name}" + (f" — {detail}" if detail else ""))
        else:
            failed += 1
            print(f"FAIL | {name}" + (f" — {detail}" if detail else ""))

    try:
        from NLP.nlp_pipeline import load_nlp_model, predict_ktas

        model, vectorizer = load_nlp_model()
        report("Load NLP classifier and TF-IDF vectorizer", True)
    except Exception as exc:
        report("Load NLP classifier and TF-IDF vectorizer", False, str(exc))
        print(f"\nNLP checks: {passed} passed, {failed} failed.")
        return False

    try:
        prediction = predict_ktas("chest pain")
        level = int(prediction["predicted_ktas"])
        valid = level in {1, 2, 3, 4, 5}
        report(
            "NLP complaint prediction returns KTAS 1–5",
            valid,
            f"prediction={level}, category={prediction.get('emergency_category')}",
        )
    except Exception as exc:
        report("NLP complaint prediction returns KTAS 1–5", False, str(exc))

    try:
        predict_ktas("   ")
        report("NLP rejects blank complaint", False, "blank input was accepted")
    except (ValueError, TypeError) as exc:
        report("NLP rejects blank complaint", True, str(exc))
    except Exception as exc:
        report("NLP rejects blank complaint", False, f"unexpected exception: {exc}")

    print(f"\nNLP checks: {passed} passed, {failed} failed.")
    return failed == 0


def run_deep_learning_checks():
    """Here I test the saved neural network and preprocessor without retraining."""
    print_heading("DEEP LEARNING — INTEGRATION CHECKS")
    passed = 0
    failed = 0

    def report(name, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"PASS | {name}" + (f" — {detail}" if detail else ""))
        else:
            failed += 1
            print(f"FAIL | {name}" + (f" — {detail}" if detail else ""))

    try:
        import numpy as np
        import pandas as pd
        from deep_learning import (
            TRIAGE_FEATURES,
            load_deep_learning_model,
            predict_ktas,
        )

        model, preprocessor, feature_names = load_deep_learning_model()
        report("Load neural network and fitted preprocessor", True)
    except Exception as exc:
        report("Load neural network and fitted preprocessor", False, str(exc))
        print(f"\nDeep Learning checks: {passed} passed, {failed} failed.")
        return False

    try:
        dataset_path = find_dataset("ktas_cleaned.csv")
        if dataset_path is None:
            raise FileNotFoundError(
                "KTAS dataset not found in either data/ or Data/."
            )
        data = pd.read_csv(dataset_path)
        missing = [column for column in TRIAGE_FEATURES if column not in data.columns]
        if missing:
            raise ValueError("Dataset is missing features: " + ", ".join(missing))
        usable = data[TRIAGE_FEATURES].dropna()
        if usable.empty:
            raise ValueError("No complete patient record is available for testing.")

        sample = usable.iloc[[0]].copy()
        predictions, probabilities = predict_ktas(model, preprocessor, sample)
        probabilities = np.asarray(probabilities)
        predictions = np.asarray(predictions).reshape(-1)

        valid_class = predictions.size == 1 and int(predictions[0]) in {1, 2, 3, 4, 5}
        report(
            "Deep-learning prediction returns KTAS 1–5",
            valid_class,
            f"prediction={predictions.tolist()}",
        )

        valid_shape = probabilities.shape == (1, 5)
        report(
            "Deep-learning probability output has five classes",
            valid_shape,
            f"shape={probabilities.shape}",
        )

        valid_probabilities = (
            valid_shape
            and np.isfinite(probabilities).all()
            and np.all(probabilities >= 0)
            and np.all(probabilities <= 1)
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        )
        report(
            "Deep-learning probabilities are finite and sum to one",
            valid_probabilities,
            f"sum={probabilities[0].sum():.6f}" if valid_shape else "",
        )

        # Here I check that the predictor supports a small batch of patient records.
        batch = usable.iloc[:2].copy()
        if len(batch) == 2:
            batch_predictions, batch_probabilities = predict_ktas(
                model, preprocessor, batch
            )
            batch_predictions = np.asarray(batch_predictions).reshape(-1)
            batch_probabilities = np.asarray(batch_probabilities)
            batch_ok = (
                batch_predictions.shape == (2,)
                and all(int(value) in {1, 2, 3, 4, 5} for value in batch_predictions)
                and batch_probabilities.shape == (2, 5)
                and np.isfinite(batch_probabilities).all()
            )
            report(
                "Deep-learning prediction supports a two-patient batch",
                batch_ok,
                f"predictions={batch_predictions.tolist()}",
            )
        else:
            report(
                "Deep-learning prediction supports a two-patient batch",
                False,
                "Dataset contains fewer than two usable records.",
            )

        # Here I verify that missing required features are rejected.
        incomplete = sample.drop(columns=[TRIAGE_FEATURES[0]])
        try:
            predict_ktas(model, preprocessor, incomplete)
            report("Deep learning rejects a missing feature", False, "input was accepted")
        except ValueError as exc:
            report("Deep learning rejects a missing feature", True, str(exc))
        except Exception as exc:
            report(
                "Deep learning rejects a missing feature",
                False,
                f"unexpected exception: {type(exc).__name__}: {exc}",
            )

        # Here I verify that an empty batch cannot silently produce a prediction.
        try:
            predict_ktas(model, preprocessor, sample.iloc[0:0])
            report("Deep learning rejects an empty batch", False, "empty input was accepted")
        except (ValueError, TypeError) as exc:
            report("Deep learning rejects an empty batch", True, str(exc))
        except Exception as exc:
            report(
                "Deep learning rejects an empty batch",
                False,
                f"unexpected exception: {type(exc).__name__}: {exc}",
            )

    except Exception as exc:
        report("Run deep-learning prediction checks", False, f"{type(exc).__name__}: {exc}")

    print(f"\nDeep Learning checks: {passed} passed, {failed} failed.")
    return failed == 0


def main():
    """Here I run integration checks for saved components without retraining."""
    print_heading("PREDICT-CARE AI — APPLICATION STARTUP")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Python interpreter: {sys.executable}")
    print("Startup mode: load existing artifacts and test; do not retrain models.")

    file_statuses = check_files()
    ml_ok = run_traditional_ml_checks()
    nlp_ok = run_nlp_checks()
    deep_learning_ok = run_deep_learning_checks()

    print_heading("STARTUP SUMMARY")
    print(f"Traditional ML integration checks: {'PASSED' if ml_ok else 'NEEDS ATTENTION'}")
    print(f"NLP integration checks:             {'PASSED' if nlp_ok else 'NEEDS ATTENTION'}")
    print(
        "Deep Learning integration checks: "
        f"{'PASSED' if deep_learning_ok else 'NEEDS ATTENTION'}"
    )
    print("No model training was started by Main.py.")
    print("File presence and successful inference do not establish clinical validity.")
    print("Prototype only: predictions are not clinical diagnoses or treatment instructions.")

    return 0 if ml_ok and nlp_ok and deep_learning_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
