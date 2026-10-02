# NLP- Natural Language Processing

This directory contains natural language processing components for the healthcare AI system.

The NLP component analyses a patient's **Chief Complaint** text and predicts:

1. The patient's **KTAS level (1–5)**.
2. The corresponding **Emergency or Non-Emergency category**.

The component is designed as a healthcare decision-support feature and does not replace professional clinical assessment.

---

## NLP Workflow

The NLP process follows these steps:

Patient Chief Complaint  
→ Text Preprocessing  
→ TF-IDF Feature Extraction  
→ Linear SVM Classification  
→ Predicted KTAS Level (1–5)  
→ Emergency / Non-Emergency Classification

KTAS levels are grouped as:

- KTAS 1–3: Emergency
- KTAS 4–5: Non-Emergency

---

## Dataset

The NLP component uses:

`Data/ktas_cleaned.csv`

The main text feature used for NLP is:

`Chief_complain`

The prediction target is:

`KTAS_expert`

Records where the Chief Complaint was marked as `MISSING` were excluded from NLP training.

After removing these records, **1,241 records** were available for the NLP component.

---

## Text Preprocessing

Patient chief complaints are cleaned before being processed by the model.

The preprocessing includes:

- Converting text to lowercase.
- Removing punctuation and special characters.
- Removing unnecessary spaces.
- Preserving useful medical terms and abbreviations.

Example:

`LBP - Low back pain`

becomes:

`lbp low back pain`

---

## Feature Extraction

The cleaned Chief Complaint text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The vectorizer uses both:

- Unigrams — individual words.
- Bigrams — combinations of two consecutive words.

This allows the model to learn patterns from terms such as:

`chest`

and phrases such as:

`chest pain`

---

## Model Comparison

Three machine-learning models were compared using the same training and testing data.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest | 53.01% | 43.35% | 53.21% | 43.36% |
| Logistic Regression | 52.21% | 44.23% | 55.01% | 44.30% |
| Linear SVM | 56.22% | 46.45% | 52.36% | 47.35% |

Linear SVM was selected for the final NLP component because it produced the highest accuracy and Macro F1 score among the three tested models.

---

## Final KTAS Model Performance

The final Linear SVM model achieved:

- Accuracy: **56.22%**
- Macro Precision: **46.45%**
- Macro Recall: **52.36%**
- Macro F1 Score: **47.35%**

The model predicts five KTAS classes:

`1, 2, 3, 4, 5`

---

## Emergency Category Performance

The predicted KTAS level is also converted into an Emergency or Non-Emergency category.

The final results were:

- Accuracy: **77.51%**
- Emergency Precision: **81.25%**
- Emergency Recall: **80.14%**
- Emergency F1 Score: **80.69%**

---

## Files

### `nlp_preprocessing.py`

Contains the text-cleaning function used to prepare Chief Complaint text.

### `train_ktas_model.py`

Loads the dataset, preprocesses the Chief Complaint text, creates TF-IDF features, trains the Linear SVM model, and saves the trained model and vectorizer.

### `evaluate_ktas_model.py`

Evaluates the trained model using classification metrics and confusion matrices for both KTAS and Emergency/Non-Emergency predictions.

### `compare_models.py`

Compares Random Forest, Logistic Regression, and Linear SVM models.

### `predict_ktas.py`

Provides a simple interface where a Chief Complaint can be entered and the trained model returns a predicted KTAS level and Emergency category.

---

## Running the NLP Component

Train the model:

```bash
python NLP/train_ktas_model.py