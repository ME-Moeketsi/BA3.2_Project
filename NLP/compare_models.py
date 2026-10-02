import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("Data/ktas_cleaned.csv")

# Remove records where the chief complaint is MISSING
df = df[
    df["Chief_complain"].astype(str).str.strip().str.upper() != "MISSING"
].copy()


# --------------------------------------------------
# DEFINE INPUT AND TARGET
# --------------------------------------------------

X = df["Chief_complain"]
y = df["KTAS_expert"]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# --------------------------------------------------
# CREATE MODELS
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

logistic_regression = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

linear_svm = LinearSVC(
    class_weight="balanced",
    random_state=42
)

models = {
    "Random Forest": random_forest,
    "Logistic Regression": logistic_regression,
    "Linear SVM": linear_svm
}


# --------------------------------------------------
# TRAIN AND EVALUATE MODELS
# --------------------------------------------------

print("KTAS NLP MODEL COMPARISON")
print("-------------------------")

for model_name, model in models.items():

    # Train model
    model.fit(X_train_tfidf, y_train)

    # Predict KTAS levels
    y_pred = model.predict(X_test_tfidf)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

# --------------------------------------------------
# DETAILED LINEAR SVM EVALUATION
# --------------------------------------------------

svm_predictions = linear_svm.predict(X_test_tfidf)

print("\n\nDETAILED LINEAR SVM RESULTS")
print("---------------------------")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        svm_predictions,
        labels=[1, 2, 3, 4, 5],
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        svm_predictions,
        labels=[1, 2, 3, 4, 5]
    )
)

print(f"\n{model_name}")
print("-" * len(model_name))

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

