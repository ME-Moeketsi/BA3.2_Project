"""
Predict-Care AI
NLP Pipeline

Here I am creating the reusable NLP component that Main.py calls.

The pipeline loads the saved Linear SVM classifier and TF-IDF
vectorizer. It checks that they are compatible and runs a small
example prediction to verify that the complete prediction path works.

Here I am keeping model training in train_ktas_model.py.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import joblib

from NLP.nlp_preprocessing import clean_text


# ============================================================
# MODEL PATHS
# ============================================================

# Here I am finding the main project directory from this file.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Here I am defining the locations of the saved NLP artifacts.
MODEL_PATH = PROJECT_DIR / "Models" / "ktas_nlp_model.pkl"
TFIDF_PATH = PROJECT_DIR / "Models" / "ktas_tfidf_vectorizer.pkl"


# ============================================================
# LOAD NLP MODEL
# ============================================================

def load_nlp_model():
    """Here I am loading the saved classifier and its TF-IDF vectorizer."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"NLP model not found: {MODEL_PATH}"
        )

    if not TFIDF_PATH.is_file():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {TFIDF_PATH}"
        )

    # Here I am loading the artifacts created by train_ktas_model.py.
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    # Here I am checking that the loaded artifacts expose the methods
    # required by this text-classification pipeline.
    if not callable(getattr(model, "predict", None)):
        raise TypeError(
            "The saved NLP model does not provide a predict() method."
        )

    if not callable(getattr(tfidf, "transform", None)):
        raise TypeError(
            "The saved TF-IDF vectorizer does not provide a transform() method."
        )

    return model, tfidf


# ============================================================
# PREDICT KTAS
# ============================================================

def _predict_with_loaded_artifacts(chief_complaint, model, tfidf):
    """
    Here I am cleaning and predicting a complaint using already-loaded
    artifacts so the pipeline does not need to load them repeatedly.
    """

    if chief_complaint is None:
        raise ValueError("Chief complaint cannot be empty.")

    chief_complaint = str(chief_complaint).strip()

    if not chief_complaint:
        raise ValueError("Chief complaint cannot be empty.")

    # Here I am applying the same text-cleaning function used in training.
    cleaned_complaint = clean_text(chief_complaint)

    if not cleaned_complaint:
        raise ValueError(
            "Chief complaint contains no usable letters or numbers."
        )

    # Here I am converting the cleaned text into the trained TF-IDF features.
    complaint_tfidf = tfidf.transform([cleaned_complaint])

    # Here I am checking that the vectorizer output matches the number
    # of features the classifier was trained to accept.
    model_feature_count = getattr(model, "n_features_in_", None)
    vectorizer_feature_count = complaint_tfidf.shape[1]

    if (
        model_feature_count is not None
        and model_feature_count != vectorizer_feature_count
    ):
        raise ValueError(
            "The saved NLP model and TF-IDF vectorizer are incompatible: "
            f"the model expects {model_feature_count} features, but the "
            f"vectorizer produced {vectorizer_feature_count}."
        )

    # Here I am asking the trained classifier to predict the KTAS level.
    predicted_ktas = int(model.predict(complaint_tfidf)[0])

    # Here I am checking that the result belongs to the expected KTAS scale.
    if predicted_ktas not in {1, 2, 3, 4, 5}:
        raise ValueError(
            f"The NLP model returned an unexpected KTAS level: {predicted_ktas}"
        )

    # Here I am mapping KTAS levels to the project's broad emergency label.
    emergency_category = (
        "Emergency" if predicted_ktas <= 3 else "Non-Emergency"
    )

    return {
        "chief_complaint": chief_complaint,
        "cleaned_complaint": cleaned_complaint,
        "predicted_ktas": predicted_ktas,
        "emergency_category": emergency_category,
    }


def predict_ktas(chief_complaint):
    """
    Here I am exposing a reusable prediction function for other project
    components without requiring them to manage model files themselves.
    """

    model, tfidf = load_nlp_model()

    return _predict_with_loaded_artifacts(
        chief_complaint,
        model,
        tfidf,
    )


# ============================================================
# NLP SYSTEM STATUS
# ============================================================

def nlp_status():
    """Here I am checking whether both saved NLP artifacts are present."""

    if MODEL_PATH.is_file() and TFIDF_PATH.is_file():
        return "READY"

    return "NOT READY"


# ============================================================
# RUN NLP PIPELINE
# ============================================================

def run_nlp_pipeline():
    """
    Here I am keeping the no-argument function expected by Main.py.

    I load both artifacts, verify their interface and feature compatibility,
    and run an example complaint through the actual prediction path.
    """

    status = nlp_status()

    print("\n")
    print("############################################")
    print("# NLP")
    print("# PATIENT CHIEF COMPLAINT PROCESSING")
    print("############################################")

    print(f"\nNLP Status: {status}")
    print("\nNLP model:")
    print("Models/ktas_nlp_model.pkl")
    print("\nTF-IDF vectorizer:")
    print("Models/ktas_tfidf_vectorizer.pkl")

    if status != "READY":
        missing_files = [
            str(path.relative_to(PROJECT_DIR))
            for path in (MODEL_PATH, TFIDF_PATH)
            if not path.is_file()
        ]
        raise FileNotFoundError(
            "Required NLP artifact(s) are missing: "
            + ", ".join(missing_files)
        )

    # Here I am loading the artifacts once for the pipeline check.
    model, tfidf = load_nlp_model()

    # Here I am checking that the classifier and vectorizer have matching
    # feature counts before I attempt a prediction.
    model_feature_count = getattr(model, "n_features_in_", None)
    vectorizer_feature_count = len(tfidf.get_feature_names_out())

    if (
        model_feature_count is not None
        and model_feature_count != vectorizer_feature_count
    ):
        raise ValueError(
            "The saved NLP model and TF-IDF vectorizer are incompatible: "
            f"the model expects {model_feature_count} features, but the "
            f"vectorizer contains {vectorizer_feature_count}."
        )

    # Here I am using an example complaint only to test the pipeline end to end.
    example_result = _predict_with_loaded_artifacts(
        "Open Wound",
        model,
        tfidf,
    )

    print("\nNLP artifacts loaded and compatibility checks passed.")
    print("Example prediction test completed.")
    print(f"Example complaint: {example_result['chief_complaint']}")
    print(f"Predicted KTAS level: {example_result['predicted_ktas']}")
    print(f"Predicted category: {example_result['emergency_category']}")
    print(
        "\nImportant: This example is a software-functionality check only. "
        "The prediction is not a clinical assessment or a validated care decision."
    )

    # Here I am returning READY to preserve the existing Main.py interface.
    return "READY"


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    run_nlp_pipeline()
