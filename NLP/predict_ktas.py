import joblib
from nlp_preprocessing import clean_text

# --------------------------------------------------
# LOAD SAVED MODEL AND TF-IDF VECTORIZER
# --------------------------------------------------

model = joblib.load("Models/ktas_nlp_model.pkl")
tfidf = joblib.load("Models/ktas_tfidf_vectorizer.pkl")


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_ktas(chief_complaint):

    # Clean the patient's chief complaint
    cleaned_complaint = clean_text(chief_complaint)

    # Convert the cleaned complaint into TF-IDF features
    complaint_tfidf = tfidf.transform([cleaned_complaint])

    # Predict KTAS level
    predicted_ktas = int(model.predict(complaint_tfidf)[0])

    # Convert KTAS level into emergency category
    if predicted_ktas <= 3:
        emergency_category = "Emergency"
    else:
        emergency_category = "Non-Emergency"

    return predicted_ktas, emergency_category


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

print("\nPREDICT-CARE AI")
print("NLP Patient Triage Support")
print("--------------------------")

chief_complaint = input(
    "\nEnter patient chief complaint: "
)


# --------------------------------------------------
# MAKE PREDICTION
# --------------------------------------------------

predicted_ktas, emergency_category = predict_ktas(
    chief_complaint
)


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print("\nPREDICTION RESULT")
print("-----------------")

print(f"Chief Complaint: {chief_complaint}")
print(f"Predicted KTAS Level: {predicted_ktas}")
print(f"Emergency Category: {emergency_category}")

print(
    "\nNote: This prediction is for decision-support only "
    "and does not replace professional clinical assessment."
)