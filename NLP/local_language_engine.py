"""
Predict-Care AI — Local Language Understanding Engine

This version does not use Ollama, a generative LLM, or a cloud API.
It uses local intent classification, rules, and fixed response templates.
It reuses the existing NLP triage pipeline when available.
"""

import re

# I am importing the tools used to train a small local intent classifier.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# I am connecting to the unified prediction pipeline.
try:
    try:
        from .pipeline import predict_ktas
    except ImportError:
        from pipeline import predict_ktas
    TRIAGE_MODEL_AVAILABLE = True
    TRIAGE_IMPORT_ERROR = None
except Exception as error:
    predict_ktas = None
    TRIAGE_MODEL_AVAILABLE = False
    TRIAGE_IMPORT_ERROR = str(error)


TRAINING_DATA = {
    "GREETING": [
        "hello", "hi", "hey", "good morning",
        "good afternoon", "good evening", "how are you"
    ],
    "HELP": [
        "help", "what can you do", "what can I ask",
        "show me your commands", "how do I use this system"
    ],
    "EXPLAIN_KTAS": [
        "what is KTAS", "explain KTAS",
        "what does KTAS stand for",
        "explain the Korean Triage and Acuity Scale",
        "explain triage in simple language"
    ],
    "PREDICT_TRIAGE": [
        "predict triage for headache",
        "predict KTAS for fever",
        "classify this chief complaint",
        "run a triage prediction for chest pain",
        "what does the model predict for dizziness",
        "classify the symptom nausea",
        "predict the triage level for abdominal pain"
    ],
    "PREDICT_WAIT": [
        "how long will I wait",
        "predict my waiting time",
        "how long until I see a doctor",
        "what is the hospital waiting time",
        "how long is the queue",
        "estimate my waiting time at the hospital"
    ],
    "PREDICT_MEDICATION_WAIT": [
        "how long will I wait for medicine",
        "I am collecting chronic medication",
        "predict pharmacy waiting time",
        "how long until I collect my medication",
        "how long will I wait to collect my pills"
    ],
    "EXPLAIN_LIMITATIONS": [
        "is this a medical diagnosis",
        "can this replace a doctor",
        "what are the model limitations",
        "is this a clinical assessment",
        "what is the difference between prediction and diagnosis"
    ],
    "GOODBYE": [
        "goodbye", "bye", "see you later", "exit", "quit"
    ],
}

RESPONSES = {
    "GREETING": (
        "Hello! I am the Predict-Care AI local assistant. "
        "You can ask about KTAS, request an experimental triage prediction, "
        "or ask about waiting-time predictions."
    ),
    "HELP": (
        "Here is what I can do:\n"
        "1. Explain KTAS.\n"
        "2. Run the existing NLP triage classifier.\n"
        "3. Recognize waiting-time requests.\n"
        "4. Explain the limitations of AI predictions.\n\n"
        "Example: /triage headache\n"
        "Type /quit to exit."
    ),
    "EXPLAIN_KTAS": (
        "KTAS stands for Korean Triage and Acuity Scale. "
        "It is a system used to prioritize patients according to urgency. "
        "A Predict-Care AI output is an experimental model prediction, "
        "not a clinical assessment."
    ),
    "PREDICT_WAIT": (
        "I recognized your waiting-time request. However, this version "
        "is not yet connected to a verified patient-specific waiting-time "
        "input schema. I will not invent a waiting-time estimate."
    ),
    "PREDICT_MEDICATION_WAIT": (
        "I recognized your medication-collection waiting-time request. "
        "The exact input features required by the existing medication-wait "
        "model must be verified before I can safely run that prediction. "
        "No waiting-time value has been generated."
    ),
    "EXPLAIN_LIMITATIONS": (
        "A prediction is an estimate produced by a machine-learning model "
        "using patterns learned from training data. A clinical assessment "
        "is performed by a qualified healthcare professional using the "
        "patient's symptoms, history, examination and other relevant "
        "information. A model prediction is not a diagnosis and must not "
        "replace professional assessment. Predict-Care AI is an academic "
        "prototype."
    ),
    "GOODBYE": (
        "Goodbye! Remember that Predict-Care AI is an academic prototype, "
        "not a medical service."
    ),
}

URGENT_RESPONSE = (
    "URGENT SAFETY NOTICE: Your message mentions symptoms that may require "
    "emergency attention. If you have chest pain with difficulty breathing, "
    "severe breathing difficulty, signs of stroke, unconsciousness, or "
    "severe bleeding, seek emergency medical help now. In South Africa, "
    "call 112 from a mobile phone or 10177 for an ambulance. Do not wait "
    "for this prototype to assess you."
)

URGENT_PATTERNS = [
    # These simple red-flag checks are deliberately broader than the trained
    # classifier and take precedence over a queue-time estimate.
    r"\b(chest pain|shortness of breath|severe difficulty breathing)\b",
    r"\bchest pain\b.{0,100}\b(difficulty breathing|can't breathe|cannot breathe|"
    r"shortness of breath|struggling to breathe)\b",
    r"\b(difficulty breathing|can't breathe|cannot breathe|"
    r"struggling to breathe|severe breathing difficulty)\b",
    r"\b(slurred speech|face drooping|face droop|sudden weakness|"
    r"stroke symptoms)\b",
    r"\b(unconscious|not responding|severe bleeding|bleeding heavily)\b",
]

URGENT_REGEX = re.compile("|".join(URGENT_PATTERNS), re.IGNORECASE)


class LocalLanguageEngine:

    def __init__(self, confidence_threshold=0.40):
        self.confidence_threshold = confidence_threshold
        self.intent_model = None

        if SKLEARN_AVAILABLE:
            sentences = []
            labels = []
            for intent, examples in TRAINING_DATA.items():
                sentences.extend(examples)
                labels.extend([intent] * len(examples))

            self.intent_model = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        strip_accents="unicode"
                    )
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ])
            self.intent_model.fit(sentences, labels)

    @staticmethod
    def normalize(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9'\s:/-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def detect_urgent_symptoms(text):
        return bool(URGENT_REGEX.search(str(text)))

    @staticmethod
    def extract_complaint(text):
        complaint = str(text).strip()

        patterns = [
            r"^\s*/(?:triage|predict)\s+",
            r"^\s*predict\s+(?:the\s+)?(?:triage|KTAS)"
            r"(?:\s+level)?\s+(?:for|of)\s+",
            r"^\s*predict\s+(?:the\s+)?(?:triage|KTAS)\s+",
            r"^\s*(?:run\s+)?(?:a\s+)?triage\s+prediction\s+for\s+",
            r"^\s*classify\s+(?:the\s+)?(?:symptom|complaint)?\s*:?\s*",
            r"^\s*(?:chief complaint|complaint)\s*:\s*",
        ]

        for pattern in patterns:
            updated = re.sub(pattern, "", complaint, flags=re.IGNORECASE)
            if updated != complaint:
                complaint = updated.strip()
                break

        return complaint.strip(" .?!")

    def classify_intent(self, text):
        message = self.normalize(text)

        if not message:
            return "UNKNOWN"

        if message in ("/help", "help"):
            return "HELP"

        if message in ("/quit", "/exit", "quit", "exit"):
            return "GOODBYE"

        explanation_phrases = (
            "diagnosis",
            "clinical assessment",
            "clinically accurate",
            "limitations",
            "replace a doctor",
            "trust the prediction",
            "difference between prediction and diagnosis",
            "difference between a prediction and diagnosis",
            "prediction versus diagnosis",
            "prediction vs diagnosis",
        )

        if any(phrase in message for phrase in explanation_phrases):
            return "EXPLAIN_LIMITATIONS"

        medication_phrases = (
            "medication",
            "medicine collection",
            "collect my medicine",
            "pharmacy",
            "chronic medication",
            "collect my pills",
        )

        if any(phrase in message for phrase in medication_phrases):
            return "PREDICT_MEDICATION_WAIT"

        wait_phrases = (
            "waiting time",
            "wait time",
            "how long will i wait",
            "hospital queue",
            "how long is the queue",
            "how long until i see a doctor",
            "if i arrive at",
            "when i arrive at",
        )

        if any(phrase in message for phrase in wait_phrases):
            return "PREDICT_WAIT"

        prediction_phrases = (
            "predict",
            "classify",
            "triage for",
            "ktas for",
            "chief complaint",
            "complaint:",
        )

        if any(phrase in message for phrase in prediction_phrases):
            return "PREDICT_TRIAGE"

        patient_context = (
            "patient",
            "he has",
            "she has",
            "they have",
            "he is",
            "she is",
            "they are",
            "has arrived",
            "has a",
            "complaining of",
            "suffering from",
            "says he",
            "says she",
            "very much in pain",
            "in pain",
        )

        symptom_terms = (
            "wound",
            "fever",
            "pain",
            "pains",
            "chest pain",
            "chest pains",
            "bleeding",
            "burn",
            "swelling",
            "injury",
            "dizzy",
            "dizziness",
            "vomiting",
            "nausea",
            "breathing",
            "unconscious",
            "unresponsive",
            "headache",
            "shortness of breath",
        )

        has_patient_context = any(
            phrase in message for phrase in patient_context
        )
        has_symptom = any(
            symptom in message for symptom in symptom_terms
        )

        if has_patient_context and has_symptom:
            return "PREDICT_TRIAGE"

        if "ktas" in message or "acuity scale" in message:
            return "EXPLAIN_KTAS"

        if (
            message in ("hello", "hi", "hey")
            or message.startswith((
                "hello ",
                "hi ",
                "hey ",
                "good morning",
                "good afternoon",
                "good evening",
            ))
        ):
            return "GREETING"

        if any(phrase in message for phrase in (
            "what can you do",
            "what can i ask",
            "show commands",
            "how do i use this",
        )):
            return "HELP"

        if any(symptom in message for symptom in symptom_terms):
            return "PREDICT_TRIAGE"

        if self.intent_model is not None:
            probabilities = self.intent_model.predict_proba([message])[0]
            best_index = probabilities.argmax()
            confidence = float(probabilities[best_index])

            if confidence >= self.confidence_threshold:
                return str(self.intent_model.classes_[best_index])

        return "UNKNOWN"

    def run_triage_prediction(self, complaint):
        critical_phrases = (
            "not breathing",
            "stopped breathing",
            "cannot breathe",
            "can't breathe",
            "unable to breathe",
            "unconscious",
            "unresponsive",
            "severe chest pain",
            "crushing chest pain",
        )

        normalized_complaint = self.normalize(complaint)

        if any(phrase in normalized_complaint for phrase in critical_phrases):
            return (
                "SAFETY ALERT: This description may indicate a "
                "medical emergency. Contact emergency services "
                "immediately. In South Africa, call 112 from a "
                "mobile phone or 10177 for an ambulance. Do not "
                "wait for a Predict-Care AI prediction."
            )

        if not complaint:
            return (
                "Please provide a chief complaint. Example: /triage headache"
            )

        if predict_ktas is None:
            return (
                "The unified prediction pipeline could not be loaded.\n"
                "Check that you are running this program from the project root "
                "and that the model files and dependencies are present.\n"
                f"Import detail: {TRIAGE_IMPORT_ERROR}"
            )

        try:
            result = predict_ktas(complaint)
            return (
                f"Existing NLP classifier output for '{complaint}': "
                f"KTAS {result['predicted_ktas']} "
                f"({result['emergency_category']}).\n\n"
                "This is an experimental model output, not a clinical "
                "assessment. It cannot guarantee that a complaint is "
                "non-urgent. Seek professional help for concerning symptoms."
            )
        except Exception as error:
            return (
                "The existing NLP prediction failed. No prediction was invented.\n"
                f"Error: {type(error).__name__}: {error}"
            )

    def respond(self, text):
        message = self.normalize(text)

        if self.detect_urgent_symptoms(text):
            return URGENT_RESPONSE, False

        if message in ("/quit", "/exit", "quit", "exit"):
            return RESPONSES["GOODBYE"], True

        if message == "/help":
            return RESPONSES["HELP"], False

        if message == "/reset":
            return "Ready for a new request. What would you like to do?", False

        if message.startswith("/triage ") or message.startswith("/predict "):
            complaint = self.extract_complaint(text)
            return self.run_triage_prediction(complaint), False

        intent = self.classify_intent(text)

        if intent == "PREDICT_TRIAGE":
            complaint = self.extract_complaint(text)

            if not complaint or complaint.lower() in (
                "this complaint", "the complaint", "triage", "ktas"
            ):
                return (
                    "I recognized a triage-prediction request. "
                    "Please include the complaint, for example: /triage headache"
                ), False

            return self.run_triage_prediction(complaint), False

        if intent in RESPONSES:
            return RESPONSES[intent], False

        return (
            "I could not confidently identify that request. "
            "Try /help, ask 'What is KTAS?', or enter /triage followed "
            "by a chief complaint."
        ), False


def main():
    assistant = LocalLanguageEngine()

    print("=" * 55)
    print("Predict-Care AI — Local Language Engine")
    print("No Ollama or cloud language model is required.")
    print(
        "Intent classifier:",
        "available" if assistant.intent_model is not None
        else "rules only (scikit-learn unavailable)"
    )
    print(
        "Unified prediction pipeline:",
        "available" if TRIAGE_MODEL_AVAILABLE else "not loaded"
    )
    print("Type /help for commands or /quit to exit.")
    print("=" * 55)

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Goodbye.")
            break

        if not user_text:
            print("Assistant: Please enter a message.")
            continue

        response, should_exit = assistant.respond(user_text)
        print(f"\nAssistant: {response}")

        if should_exit:
            break


if __name__ == "__main__":
    main()