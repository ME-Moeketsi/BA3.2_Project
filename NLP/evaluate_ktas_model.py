import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from nlp_preprocessing import clean_text


# ----- LOAD DATASET -----

df = pd.read_csv("Data/ktas_cleaned.csv")

# Remove missing chief complaints
df = df[
    df["Chief_complain"].astype(str).str.strip().str.upper() != "MISSING"
].copy()

df["cleaned_complaint"] = df["Chief_complain"].apply(clean_text)


# ----- SET INPUT AND TARGET -----

X = df["cleaned_complaint"]
y = df["KTAS_expert"]


# ----- SPLIT DATA FOR TRAINING AND TESTING -----

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ----- LOAD TRAINED MODEL AND TF-IDF VECTORIZER -----

model = joblib.load("Models/ktas_nlp_model.pkl")
tfidf = joblib.load("Models/ktas_tfidf_vectorizer.pkl")


# ----- TRANSFORM TEST COMPLAINTS -----

X_test_tfidf = tfidf.transform(X_test)


# ----- MAKE PREDICTIONS -----

y_pred = model.predict(X_test_tfidf)


# ----- EVALUATE KTAS PREDICTIONS -----

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


print("KTAS NLP MODEL EVALUATION")
print("-------------------------")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# ----- RESULTS FOR EACH KTAS LEVEL -----

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4, 5],
        zero_division=0
    )
)


# ----- KTAS CONFUSION MATRIX -----

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5]
)

print(cm)


# ----- EMERGENCY AND NON-EMERGENCY EVALUATION -----

# Convert KTAS predictions into emergency categories
predicted_binary = [
    "Emergency" if ktas <= 3 else "Non-Emergency"
    for ktas in y_pred
]

# Get the actual emergency categories
actual_binary = df.loc[y_test.index, "KTAS_binary"]


# ----- CALCULATE EMERGENCY CATEGORY METRICS -----

binary_accuracy = accuracy_score(
    actual_binary,
    predicted_binary
)

binary_precision = precision_score(
    actual_binary,
    predicted_binary,
    pos_label="Emergency",
    zero_division=0
)

binary_recall = recall_score(
    actual_binary,
    predicted_binary,
    pos_label="Emergency",
    zero_division=0
)

binary_f1 = f1_score(
    actual_binary,
    predicted_binary,
    pos_label="Emergency",
    zero_division=0
)


print("\nEMERGENCY CATEGORY EVALUATION")
print("-----------------------------")

print(f"\nAccuracy:  {binary_accuracy:.4f}")
print(f"Precision: {binary_precision:.4f}")
print(f"Recall:    {binary_recall:.4f}")
print(f"F1 Score:  {binary_f1:.4f}")


print("\nEmergency Classification Report:")

print(
    classification_report(
        actual_binary,
        predicted_binary,
        labels=["Emergency", "Non-Emergency"],
        zero_division=0
    )
)


# ----- EMERGENCY CATEGORY CONFUSION MATRIX -----

print("Emergency Confusion Matrix:")

binary_cm = confusion_matrix(
    actual_binary,
    predicted_binary,
    labels=["Emergency", "Non-Emergency"]
)

print(binary_cm)