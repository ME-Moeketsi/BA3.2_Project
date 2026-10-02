from nlp_preprocessing import clean_text

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("Data/ktas_cleaned.csv")


# --------------------------------------------------
# REMOVE MISSING CHIEF COMPLAINTS
# --------------------------------------------------

df = df[
    df["Chief_complain"].astype(str).str.strip().str.upper() != "MISSING"
].copy()


# Clean the chief complaint text
df["cleaned_complaint"] = df["Chief_complain"].apply(clean_text)


# --------------------------------------------------
# DEFINE INPUT AND TARGET
# --------------------------------------------------

X = df["Chief_complain"]
y = df["KTAS_expert"]
X = df["cleaned_complaint"]

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
# DISPLAY RESULTS
# --------------------------------------------------

print("NLP Training Data Prepared")

print("\nTotal usable records:")
print(len(df))

print("\nTraining records:")
print(len(X_train))

print("\nTesting records:")
print(len(X_test))

print("\nTraining KTAS distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting KTAS distribution:")
print(y_test.value_counts().sort_index())

print("\nNumber of TF-IDF features:")
print(len(tfidf.get_feature_names_out()))

print("\nExample TF-IDF features:")
print(tfidf.get_feature_names_out()[:30])


# --------------------------------------------------
# TRAIN RANDOM FOREST CLASSIFIER
# --------------------------------------------------

from sklearn.svm import LinearSVC

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)


model.fit(X_train_tfidf, y_train)

print("\nKTAS NLP model training completed successfully.")

# --------------------------------------------------
# SAVE MODEL AND TF-IDF VECTORIZER
# --------------------------------------------------

import joblib

joblib.dump(model, "Models/ktas_nlp_model.pkl")
joblib.dump(tfidf, "Models/ktas_tfidf_vectorizer.pkl")

print("KTAS NLP model saved successfully.")
print("TF-IDF vectorizer saved successfully.")