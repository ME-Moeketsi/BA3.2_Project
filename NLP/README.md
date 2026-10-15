# NLP – Natural Language Processing

## 1. Overview

This folder contains the Natural Language Processing (NLP) part of our **Predict-Care AI** project.

NLP allows a computer to understand and work with human language.

Our NLP component helps predict how urgent a patient's condition might be based on their **chief complaint** (the main reason they are seeking medical help).

For example, a patient might say:

"I have severe chest pain."

The system analyses the complaint and predicts:

- **KTAS Level (1–5):** A level used to describe how urgently a patient may need medical attention.
- **Emergency Category:** Emergency or Non-Emergency.

The system supports healthcare workers when making decisions. It does not replace a doctor, nurse, or professional medical assessment.

---

## 2. How the NLP System Works

The system allows users to enter a patient's complaint in two ways.

### Option 1: Text Input

The user types the patient's complaint using the keyboard.

Example:

`I have a headache`

### Option 2: Speech Input

The user speaks the patient's complaint into a microphone.

The system records the voice and converts it into written text using speech recognition.

Example:

**Patient says:** "I have severe abdominal pain."

**System converts it to:** `I have severe abdominal pain`

After receiving the complaint, the system uses the trained NLP model to predict the KTAS level and emergency category.

### Complete Process

1. The user chooses between text input and speech input.
2. The user types or speaks the patient's complaint.
3. If speech is selected, the system converts the voice into text.
4. The system cleans the complaint text.
5. The trained NLP model analyses the complaint.
6. The system predicts a KTAS level from 1 to 5.
7. The system displays the Emergency or Non-Emergency category.

### KTAS Levels

| KTAS Level | Emergency Category |
|---|---|
| 1 | Emergency |
| 2 | Emergency |
| 3 | Emergency |
| 4 | Non-Emergency |
| 5 | Non-Emergency |

These categories are used in our project to simplify the model's results.

---

## 3. Dataset Used

The NLP component uses the following dataset:

`Data/ktas_cleaned.csv`

Two important columns are:

- **Chief_complain:** Contains the patient's main complaint.
- **KTAS_expert:** Contains the KTAS level assigned by an expert.

We use the patient complaints to train the model to recognise patterns linked to different KTAS levels.

Records with missing complaint information are removed before training.

The original model comparison used **1,241 usable records**. After the dataset was updated, a later training run used **1,246 usable records**.

---

## 4. Preparing the Text

Before the model analyses a complaint, the text is cleaned.

This includes:

- Changing capital letters to lowercase.
- Removing unnecessary punctuation.
- Removing extra spaces.
- Keeping useful medical words and abbreviations.

**Example:**

Before cleaning:

`LBP - Low back pain`

After cleaning:

`lbp low back pain`

This helps the model process complaints in a consistent way.

---

## 5. Training the NLP Model

We trained the NLP model using patient complaints from the dataset.

The system uses **TF-IDF** to convert words into numbers that a machine-learning model can understand.

It looks at individual words and combinations of two words.

For example:

- `chest`
- `chest pain`

We compared three machine-learning models:

1. Random Forest
2. Logistic Regression
3. Linear SVM

**Linear SVM** performed best in our original comparison, so we selected it for the KTAS prediction component.

---

## 6. Model Results

The following results are from our original model evaluation.

### KTAS Level Prediction

| Model | Accuracy |
|---|---:|
| Random Forest | 53.01% |
| Logistic Regression | 52.21% |
| Linear SVM | 56.22% |

The selected Linear SVM model correctly predicted the exact KTAS level for approximately **56 out of every 100 test cases**.

### Emergency Category Prediction

The original model achieved **77.51% accuracy** when predicting whether a case belonged to the Emergency or Non-Emergency category.

**Important:** These results were obtained before the latest dataset update. The updated model must be evaluated again to confirm its current accuracy.

These results also show that the model can make mistakes. It is a student project and must not be used to make real patient-care decisions without proper clinical validation.

---

## 7. Speech Recognition

We added speech recognition so that users do not always have to type patient complaints.

The speech recognition component:

1. Uses a microphone to record the user's voice.
2. Records audio for 10 seconds.
3. Converts the recorded speech into text.
4. Sends the text to the existing NLP model.
5. Displays the predicted KTAS level and emergency category.

Speech recognition uses the Python libraries **SpeechRecognition** and **PyAudio**.

The current version uses Google's online speech recognition service, which requires an internet connection.

For privacy, only fictional patient complaints should be used during testing because audio is sent to an external service.

---

## 8. Files in the NLP Folder

| File | Purpose |
|---|---|
| `nlp_preprocessing.py` | Cleans the patient's complaint text. |
| `compare_models.py` | Compares different machine-learning models. |
| `train_ktas_model.py` | Trains the KTAS prediction model and saves it. |
| `evaluate_ktas_model.py` | Checks how well the trained model performs. |
| `predict_ktas.py` | Allows a user to type a complaint and receive a prediction. |
| `ktas_speech_input.py` | Records a spoken complaint and uses the NLP model to make a prediction. |
| `ktas_input_menu.py` | Allows users to choose between text and speech input. |

### Saved Model Files

The trained NLP model uses two files:

- `Models/ktas_nlp_model.pkl` — The trained prediction model.
- `Models/ktas_tfidf_vectorizer.pkl` — Converts complaint text into numerical features.

These files are generated when the model is trained.

---

## 9. How to Run the NLP Component

Open the project in VS Code and use the terminal from the main project folder.

Make sure the correct Python environment is active and the required libraries are installed.

### Train the Model

```bash
python NLP/train_ktas_model.py
```

This trains the model using the dataset and saves the required model files.

### Test Text Input Only

```bash
python NLP/predict_ktas.py
```

This allows the user to type a complaint and receive a prediction.

### Test Speech Input Only

```bash
python NLP/ktas_speech_input.py
```

This records the user's voice, converts it into text, and makes a prediction.

### Test Both Text and Speech Input

**Recommended for demonstrating the completed NLP component:**

```bash
python NLP/ktas_input_menu.py
```

The system displays:

```text
PREDICT-CARE AI
NLP Patient Triage Support

Choose input method:
1. Enter complaint using text
2. Enter complaint using speech
3. Exit
```

The user selects their preferred input method.

The microphone device may need to be changed when running the speech component on a different computer.

---

## 10. Example Test Results

The following examples were tested successfully using speech input.

### Example 1: Headache

**Recognized complaint:** I have a headache

**Model prediction:**
- KTAS Level: 4
- Emergency Category: Non-Emergency

### Example 2: Severe Abdominal Pain

**Recognized complaint:** I have severe abdominal pain

**Model prediction:**
- KTAS Level: 3
- Emergency Category: Emergency

These examples show that speech recognition and the NLP prediction model are connected and working.

They do not prove that the predictions are medically correct.

---

## 11. Current Progress

The following features have been implemented and tested successfully:

- [x] Cleaning patient complaint text.
- [x] Training the KTAS prediction model.
- [x] Comparing machine-learning models.
- [x] Predicting KTAS levels 1–5.
- [x] Predicting Emergency or Non-Emergency categories.
- [x] Accepting complaints through keyboard input.
- [x] Recording complaints using a microphone.
- [x] Converting speech into text.
- [x] Allowing users to choose between text and speech input.

### Next Steps

- Evaluate the model using the latest updated dataset.
- Integrate the NLP component into the main Predict-Care AI user interface (GUI).
- Continue development of NLP features for patient waiting-time analysis.

---

## 12. Important Note

Predict-Care AI is a student project designed to explore how artificial intelligence can support healthcare operations.

The NLP component provides predictions based on patterns learned from the dataset.

**It does not replace professional medical assessment, and its predictions should not be used alone to make real clinical decisions.**