from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from nlp_preprocessing import clean_text


# Here I am using project-relative paths so the evaluator works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "ktas_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "Models" / "ktas_nlp_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "Models" / "ktas_tfidf_vectorizer.pkl"
SPLIT_PATH = PROJECT_ROOT / "Models" / "ktas_nlp_split.json"
RESULTS_DIR = PROJECT_ROOT / "Results"
KTAS_LABELS = [1, 2, 3, 4, 5]
BINARY_LABELS = ["Emergency", "Non-Emergency"]


def load_and_clean_dataset():
    """Here I am applying the same validation and cleaning used by the trainer."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = {"Chief_complain", "KTAS_expert"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    complaints = df["Chief_complain"].astype("string").str.strip()
    valid_complaints = (
        complaints.notna()
        & complaints.ne("")
        & complaints.str.upper().ne("MISSING")
    )
    labels = pd.to_numeric(df["KTAS_expert"], errors="coerce")
    valid_labels = labels.notna() & labels.isin(KTAS_LABELS)

    df = df.loc[valid_complaints & valid_labels, ["Chief_complain"]].copy()
    df["KTAS_expert"] = labels.loc[df.index].astype(int)
    df["cleaned_complaint"] = df["Chief_complain"].map(clean_text)
    df = df[df["cleaned_complaint"].str.strip().ne("")].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("No usable records remain after validating the dataset.")
    return df


def save_outputs(summary_text, report_dict, ktas_cm, binary_cm):
    """Here I am saving the summary and confusion matrices for the project results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "nlp_evaluation_summary.txt").write_text(summary_text, encoding="utf-8")
    pd.DataFrame(report_dict).transpose().to_csv(
        RESULTS_DIR / "nlp_ktas_classification_report.csv", index=True, index_label="class"
    )
    pd.DataFrame(
        ktas_cm,
        index=[f"Actual KTAS {label}" for label in KTAS_LABELS],
        columns=[f"Predicted KTAS {label}" for label in KTAS_LABELS],
    ).to_csv(RESULTS_DIR / "nlp_ktas_confusion_matrix.csv")
    pd.DataFrame(
        binary_cm,
        index=[f"Actual {label}" for label in BINARY_LABELS],
        columns=[f"Predicted {label}" for label in BINARY_LABELS],
    ).to_csv(RESULTS_DIR / "nlp_emergency_confusion_matrix.csv")


def main():
    print("PREDICT-CARE AI - NLP MODEL EVALUATION")
    print("======================================")

    required_files = [MODEL_PATH, VECTORIZER_PATH, SPLIT_PATH]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_list = "\n".join(str(path.relative_to(PROJECT_ROOT)) for path in missing_files)
        raise FileNotFoundError(
            "Required NLP artifacts are missing:\n"
            f"{missing_list}\n\nRun python -u NLP/train_ktas_model.py first."
        )

    df = load_and_clean_dataset()
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    current_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if manifest.get("dataset_sha256") != current_hash:
        raise ValueError(
            "The dataset has changed since training. Retrain the model before evaluating it."
        )
    if manifest.get("usable_record_count") != len(df):
        raise ValueError(
            "The usable record count differs from the training manifest. Retrain the model."
        )

    train_indices = manifest.get("train_indices", [])
    test_indices = manifest.get("test_indices", [])
    if not train_indices or not test_indices:
        raise ValueError("The saved split manifest is incomplete. Retrain the model.")
    if max(train_indices + test_indices) >= len(df) or min(train_indices + test_indices) < 0:
        raise ValueError("Split indices do not match the current dataset. Retrain the model.")
    if set(train_indices).intersection(test_indices):
        raise ValueError("The saved train/test split overlaps. Retrain the model.")

    train_text = set(df.iloc[train_indices]["cleaned_complaint"])
    test_text = set(df.iloc[test_indices]["cleaned_complaint"])
    if train_text.intersection(test_text):
        raise ValueError(
            "Duplicate cleaned complaint text appears in both splits. "
            "The saved split is not suitable for this leakage check; retrain the model."
        )

    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    if not hasattr(model, "predict") or not hasattr(tfidf, "transform"):
        raise TypeError("The saved model or vectorizer does not expose the required methods.")
    if getattr(model, "n_features_in_", None) != len(tfidf.get_feature_names_out()):
        raise ValueError("Model feature count does not match the saved TF-IDF vectorizer.")

    X_test = df.iloc[test_indices]["cleaned_complaint"]
    y_test = df.iloc[test_indices]["KTAS_expert"]
    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    ktas_accuracy = accuracy_score(y_test, y_pred)
    ktas_precision = precision_score(y_test, y_pred, labels=KTAS_LABELS, average="macro", zero_division=0)
    ktas_recall = recall_score(y_test, y_pred, labels=KTAS_LABELS, average="macro", zero_division=0)
    ktas_f1 = f1_score(y_test, y_pred, labels=KTAS_LABELS, average="macro", zero_division=0)
    ktas_report_text = classification_report(
        y_test, y_pred, labels=KTAS_LABELS, zero_division=0
    )
    ktas_report_dict = classification_report(
        y_test, y_pred, labels=KTAS_LABELS, output_dict=True, zero_division=0
    )
    ktas_cm = confusion_matrix(y_test, y_pred, labels=KTAS_LABELS)

    # Here I am deriving actual and predicted emergency categories from the same KTAS rule.
    actual_binary = ["Emergency" if int(value) <= 3 else "Non-Emergency" for value in y_test]
    predicted_binary = ["Emergency" if int(value) <= 3 else "Non-Emergency" for value in y_pred]
    binary_accuracy = accuracy_score(actual_binary, predicted_binary)
    binary_precision = precision_score(actual_binary, predicted_binary, pos_label="Emergency", zero_division=0)
    binary_recall = recall_score(actual_binary, predicted_binary, pos_label="Emergency", zero_division=0)
    binary_f1 = f1_score(actual_binary, predicted_binary, pos_label="Emergency", zero_division=0)
    binary_report_text = classification_report(
        actual_binary, predicted_binary, labels=BINARY_LABELS, zero_division=0
    )
    binary_cm = confusion_matrix(actual_binary, predicted_binary, labels=BINARY_LABELS)

    summary_lines = [
        "PREDICT-CARE AI - NLP MODEL EVALUATION",
        "======================================",
        f"Dataset: {DATA_PATH.relative_to(PROJECT_ROOT)}",
        f"Valid records evaluated: {len(df)}",
        f"Training records recorded in manifest: {len(train_indices)}",
        f"Held-out test records: {len(test_indices)}",
        f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}",
        f"TF-IDF vectorizer: {VECTORIZER_PATH.relative_to(PROJECT_ROOT)}",
        "Split: StratifiedGroupKFold; identical cleaned complaint text is grouped",
        "",
        "FIVE-CLASS KTAS EVALUATION",
        "--------------------------",
        f"Accuracy:        {ktas_accuracy:.4f}",
        f"Macro precision: {ktas_precision:.4f}",
        f"Macro recall:    {ktas_recall:.4f}",
        f"Macro F1 score:  {ktas_f1:.4f}",
        "",
        "Classification report (KTAS 1 through 5):",
        ktas_report_text,
        "Confusion matrix (rows = actual, columns = predicted; KTAS 1 through 5):",
        str(ktas_cm),
        "",
        "EMERGENCY CATEGORY EVALUATION",
        "-----------------------------",
        f"Accuracy:  {binary_accuracy:.4f}",
        f"Precision: {binary_precision:.4f}",
        f"Recall:    {binary_recall:.4f}",
        f"F1 score:  {binary_f1:.4f}",
        "",
        "Classification report:",
        binary_report_text,
        "Confusion matrix (rows = actual, columns = predicted; Emergency first):",
        str(binary_cm),
        "",
        "IMPORTANT: These results describe this held-out dataset split only. This academic prototype must not be used to make clinical decisions.",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    print(summary_text)
    save_outputs(summary_text, ktas_report_dict, ktas_cm, binary_cm)

    print("Evaluation files saved:")
    print("  Summary: Results\\nlp_evaluation_summary.txt")
    print("  KTAS report: Results\\nlp_ktas_classification_report.csv")
    print("  KTAS confusion matrix: Results\\nlp_ktas_confusion_matrix.csv")
    print("  Emergency confusion matrix: Results\\nlp_emergency_confusion_matrix.csv")


if __name__ == "__main__":
    main()
