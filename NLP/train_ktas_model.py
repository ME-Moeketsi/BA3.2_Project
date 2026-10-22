"""
Predict-Care AI
NLP Model Training

Here I am training the NLP component that predicts the expert
KTAS level from a patient's chief complaint.

I use two complementary TF-IDF views of the cleaned complaint:

  1. Word-level TF-IDF, ngram_range=(1, 2) — captures complete
     phrases such as "chest pain" and "shortness of breath".
  2. Character-level TF-IDF (char_wb), ngram_range=(2, 4) —
     captures sub-word units so typos and paraphrases still share
     features with the canonical complaint.

The two views are combined with FeatureUnion. The classifier is a
class-balanced Linear SVM.

The held-out evaluation is group-aware: identical cleaned
complaints are kept in a single split so no exact-phrase
duplication leaks between train and test.

Every artefact and the exact split are saved so the evaluation is
fully reproducible from the source CSV.
"""

from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from nlp_preprocessing import clean_text


# ============================================================
# PROJECT PATHS AND CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "ktas_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "Models"
MODEL_PATH = MODEL_DIR / "ktas_nlp_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "ktas_tfidf_vectorizer.pkl"
SPLIT_PATH = MODEL_DIR / "ktas_nlp_split.json"

RANDOM_STATE = 42
N_SPLITS = 5
EXPECTED_KTAS_CLASSES = {1, 2, 3, 4, 5}


PARAPHRASE_PROBES = [
    "chest tightness",
    "heart attack",
    "short of breath",
    "stomache ache",
    "puking",
    "unconscious",
    "high temperature",
    "back ache",
    "head pain",
    "severe bleeding",
]


# ============================================================
# DATASET LOADING
# ============================================================

def load_and_clean_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = {"Chief_complain", "KTAS_expert"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    complaints = df["Chief_complain"].astype("string").str.strip()
    valid_complaints = (
        complaints.notna()
        & complaints.ne("")
        & complaints.str.upper().ne("MISSING")
    )

    labels = pd.to_numeric(df["KTAS_expert"], errors="coerce")
    valid_labels = labels.notna() & labels.isin(EXPECTED_KTAS_CLASSES)

    df = df.loc[valid_complaints & valid_labels, ["Chief_complain"]].copy()
    df["KTAS_expert"] = labels.loc[df.index].astype(int)
    df["cleaned_complaint"] = df["Chief_complain"].map(clean_text)

    df = df[df["cleaned_complaint"].str.strip().ne("")].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("No usable records remain after validating the dataset.")

    class_counts = df["KTAS_expert"].value_counts()
    missing_classes = EXPECTED_KTAS_CLASSES - set(class_counts.index.astype(int))
    if missing_classes:
        raise ValueError(
            f"The usable dataset is missing KTAS classes: {sorted(missing_classes)}"
        )
    if class_counts.min() < N_SPLITS:
        raise ValueError(
            f"Each KTAS class needs at least {N_SPLITS} records for this split; "
            f"class counts are {class_counts.sort_index().to_dict()}"
        )

    return df


# ============================================================
# VECTORIZER
# ============================================================

def build_vectorizer():
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        min_df=1,
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        lowercase=True,
        sublinear_tf=True,
        min_df=1,
    )

    return FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
        ]
    )


# ============================================================
# EVALUATION HELPERS
# ============================================================

def report_paraphrase_probes(model, vectorizer):
    print("\nPARAPHRASE PROBES (not from training data)")
    print("------------------------------------------")

    for probe in PARAPHRASE_PROBES:
        cleaned = clean_text(probe)
        if not cleaned:
            print(f"  {probe!r} -> (empty after cleaning)")
            continue
        try:
            features = vectorizer.transform([cleaned])
            prediction = int(model.predict(features)[0])
            print(f"  {probe!r:>28} -> cleaned={cleaned!r:>28} -> KTAS {prediction}")
        except Exception as exc:
            print(f"  {probe!r} -> ERROR: {exc}")


# ============================================================
# MAIN TRAINING WORKFLOW
# ============================================================

def main():
    print("PREDICT-CARE AI - NLP MODEL TRAINING")
    print("====================================")

    df = load_and_clean_dataset()
    X = df["cleaned_complaint"]
    y = df["KTAS_expert"]
    groups = df["cleaned_complaint"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    overlap = set(X_train.unique()).intersection(set(X_test.unique()))
    if overlap:
        raise RuntimeError(
            "Duplicate cleaned complaints were found across train and test splits."
        )

    print(f"\nDataset: {DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Valid records: {len(df)}")
    print(f"Training records: {len(train_indices)}")
    print(f"Held-out test records: {len(test_indices)}")
    print(f"Unique cleaned complaints: {df['cleaned_complaint'].nunique()}")
    print("\nTraining KTAS distribution:")
    print(y_train.value_counts().sort_index().to_string())
    print("\nHeld-out KTAS distribution:")
    print(y_test.value_counts().sort_index().to_string())

    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nWord-level TF-IDF features : {len(vectorizer.transformer_list[0][1].get_feature_names_out())}")
    print(f"Char-level TF-IDF features : {len(vectorizer.transformer_list[1][1].get_feature_names_out())}")
    print(f"Total combined features    : {X_train_tfidf.shape[1]}")

    model = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print("\nHELD-OUT TRAINING CHECK (not a substitute for independent validation)")
    print("--------------------------------------------------------------------")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(
        f"Macro F1 : "
        f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
    )
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[1, 2, 3, 4, 5],
            zero_division=0,
        )
    )

    report_paraphrase_probes(model, vectorizer)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    manifest = {
        "dataset_sha256": dataset_hash,
        "usable_record_count": int(len(df)),
        "random_state": RANDOM_STATE,
        "split_method": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "group_column": "cleaned_complaint",
        "vectorizer": "FeatureUnion(word_tfidf, char_wb_tfidf)",
        "classifier": "LinearSVC(class_weight='balanced')",
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }
    SPLIT_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nSaved artifacts:")
    print(f"Model:             {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"TF-IDF vectorizer: {VECTORIZER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Split manifest:    {SPLIT_PATH.relative_to(PROJECT_ROOT)}")
    print("\nTraining completed.")


if __name__ == "__main__":
    main()