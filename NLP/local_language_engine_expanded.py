
"""
Predict-Care AI — Local Language Understanding Engine

This version does not use Ollama, a generative LLM, or a cloud API.
It uses local intent classification, rules, and fixed response templates.
It reuses the existing NLP triage pipeline when available.
"""

import re


# Import the machine-learning libraries required for intent classification.
# If scikit-learn is unavailable, the engine can still use its predefined rules.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True

except ImportError:
    SKLEARN_AVAILABLE = False


# Connect to the existing KTAS prediction pipeline.
# The language engine remains available even if the pipeline cannot be loaded.
try:
    from pipeline import predict_ktas

    TRIAGE_MODEL_AVAILABLE = True
    TRIAGE_IMPORT_ERROR = None

except Exception as error:
    predict_ktas = None
    TRIAGE_MODEL_AVAILABLE = False
    TRIAGE_IMPORT_ERROR = str(error)


# Training examples for the eight supported conversational intents.
# Each category contains different ways a user might express the same request.
# These examples are used to train the TF-IDF and Logistic Regression classifier.
TRAINING_DATA = {
    "GREETING": [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "how are you", "hello there", "hi there",
        "hey assistant", "good day", "morning", "afternoon",
        "evening", "greetings", "howzit", "sawubona", "dumelang",
        "hello predict care", "hi predict care", "nice to meet you",
        "is anyone there", "are you available", "can we chat",
        "good morning assistant",
    ],

    "HELP": [
        "help", "what can you do", "what can I ask",
        "show me your commands", "how do I use this system",
        "what services do you offer", "show available features",
        "what are your functions", "I need help using the assistant",
        "guide me through this system", "how does predict care work",
        "what questions can I ask you", "give me some examples",
        "show me the options", "how can you assist me",
        "tell me what you support", "what is this chatbot for",
        "how do I start", "list your commands", "I am new here",
        "what can this tool help with", "show help menu",
        "explain your features", "what tasks can you perform",
        "what should I type",
    ],

    "EXPLAIN_KTAS": [
        "what is KTAS", "explain KTAS", "what does KTAS stand for",
        "explain the Korean Triage and Acuity Scale",
        "explain triage in simple language", "how does KTAS work",
        "what are the five KTAS levels", "what is a triage level",
        "what does KTAS level one mean",
        "what does KTAS level two mean",
        "what does KTAS level three mean",
        "what does KTAS level four mean",
        "what does KTAS level five mean",
        "what is patient prioritisation",
        "why do hospitals use triage",
        "what is the purpose of KTAS",
        "tell me about the triage scale",
        "how are patients prioritised",
        "what is the Korean triage scale",
        "what is the difference between KTAS levels",
        "how is urgency classified", "what does acuity mean",
        "explain emergency triage", "teach me about KTAS",
        "what does triage mean",
        "could you explain the five triage urgency categories",
        "explain the five triage levels",
        "what are the triage urgency categories",
        "describe the triage priority levels",
        "how many triage levels are there",
        "what do the five triage categories mean",
        "explain hospital triage categories",
        "tell me about patient acuity levels",
    ],

    "PREDICT_TRIAGE": [
        "predict triage for headache", "predict KTAS for fever",
        "classify this chief complaint",
        "run a triage prediction for chest pain",
        "what does the model predict for dizziness",
        "classify the symptom nausea",
        "predict the triage level for abdominal pain",
        "assess the urgency of a headache",
        "estimate the KTAS category for vomiting",
        "patient complains of back pain",
        "my patient has a fever",
        "she has a severe headache",
        "he is experiencing dizziness",
        "they have abdominal pain",
        "classify a patient with an open wound",
        "evaluate this complaint of nausea",
        "which KTAS level fits a cough",
        "predict urgency for shortness of breath",
        "can you classify this symptom",
        "I need a triage prediction",
        "check the urgency of this complaint",
        "the patient reports chest discomfort",
        "triage assessment for a head injury",
        "what triage category for a burn",
        "run the KTAS classifier for fever",
        "my patient is experiencing abdominal discomfort",
        "a patient reports stomach discomfort",
        "the patient has abdominal discomfort",
        "my patient feels discomfort in the abdomen",
        "patient complains of abdominal discomfort",
        "assess the patient with stomach discomfort",
        "triage a patient with abdominal discomfort",
        "my patient reports stomach cramps",
    ],

    "PREDICT_WAIT": [
        "how long will I wait",
        "predict my waiting time",
        "how long until I see a doctor",
        "what is the hospital waiting time",
        "how long is the queue",
        "estimate my waiting time at the hospital",
        "how much time before a doctor sees me",
        "when will the doctor attend to me",
        "estimate the doctor waiting time",
        "how long before my consultation",
        "what is my expected doctor wait",
        "how long will the patient wait to be seen",
        "calculate the doctor queue time",
        "how many minutes until a consultation",
        "when can I see the physician",
        "how much time will I spend in hospital",
        "estimate my total time in hospital",
        "what is my expected hospital stay",
        "how long will the whole hospital visit take",
        "predict the total hospital duration",
        "how long before I leave the hospital",
        "give me both waiting time estimates",
        "show doctor waiting time and total hospital time",
        "what is the predicted consultation delay",
        "how long is the patient flow process",
        "approximately when will the physician attend to me",
        "when will a physician attend to me",
        "when will the medical practitioner see me",
        "how soon will I be attended to",
        "how long until the physician sees me",
        "when can I expect to see the doctor",
        "how much longer before my consultation",
        "could you estimate my hospital queue duration",
    ],

    "PREDICT_MEDICATION_WAIT": [
        "how long will I wait for medicine",
        "I am collecting chronic medication",
        "predict pharmacy waiting time",
        "how long until I collect my medication",
        "how long will I wait to collect my pills",
        "when can I pick up my prescription",
        "how long is the pharmacy queue",
        "estimate my medication collection time",
        "how long until my tablets are ready",
        "when will the pharmacy serve me",
        "predict the time to receive medicine",
        "how much longer for my prescription",
        "I am waiting for my medication",
        "when can I collect my chronic pills",
        "how long does dispensing take",
        "what is the medication queue time",
        "estimate the wait for medicine pickup",
        "is my prescription ready yet",
        "how long before I get my drugs from pharmacy",
        "time needed for pharmacy collection",
        "how long will I wait at the dispensary",
        "how long until the pharmacist calls me",
        "what is the estimated pharmacy delay",
        "how long to get my prescription filled",
        "medication collection waiting period",
    ],

    "EXPLAIN_LIMITATIONS": [
        "is this a medical diagnosis",
        "can this replace a doctor",
        "what are the model limitations",
        "is this a clinical assessment",
        "what is the difference between prediction and diagnosis",
        "how accurate are your predictions",
        "can I trust this result",
        "is the AI clinically validated",
        "should I rely on this for treatment",
        "is this system safe for real patients",
        "does this chatbot give medical advice",
        "what if the model makes a mistake",
        "can the model be wrong",
        "who should confirm the triage result",
        "is this only a student prototype",
        "are the predictions guaranteed",
        "what data was the model trained on",
        "what are the risks of AI triage",
        "can this diagnose my illness",
        "is the prediction a final decision",
        "should a nurse review the result",
        "can AI replace medical professionals",
        "what are the limitations of predict care",
        "does this count as professional advice",
        "how reliable is the triage classification",
        "how reliable are these AI generated results",
        "can the artificial intelligence make errors",
        "should I trust the AI prediction",
    ],

    "GOODBYE": [
        "goodbye", "bye", "see you later", "exit", "quit",
        "good night", "talk to you later", "thanks goodbye",
        "bye for now", "see you soon", "I am done",
        "end chat", "close conversation", "stop chatting",
        "I have to go", "catch you later", "farewell",
        "that is all thanks", "end session", "until next time",
        "I am leaving now", "have a nice day",
        "see you tomorrow", "goodbye assistant",
        "quit the assistant",
    ],
}


# Fixed response templates for supported conversational intents.
# These responses do not generate clinical decisions or invented predictions.
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
        "Medication waiting-time prediction is not currently supported "
        "by this component. No waiting-time value has been generated."
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


# Safety message displayed when potentially life-threatening symptoms
# are detected in a user's request.
URGENT_RESPONSE = (
    "URGENT SAFETY NOTICE: Your message mentions symptoms that may require "
    "emergency attention. If you have chest pain with difficulty breathing, "
    "severe breathing difficulty, signs of stroke, unconsciousness, or "
    "severe bleeding, seek emergency medical help now. In South Africa, "
    "call 112 from a mobile phone or 10177 for an ambulance. Do not wait "
    "for this prototype to assess you."
)


# Regular expressions for recognizing selected emergency warning signs.
# This list is not comprehensive and does not replace clinical screening.
URGENT_PATTERNS = [
    r"\bchest pain\b.{0,100}\b(difficulty breathing|can't breathe|cannot breathe|shortness of breath|struggling to breathe)\b",
    r"\b(difficulty breathing|can't breathe|cannot breathe|struggling to breathe|severe breathing difficulty)\b",
    r"\b(slurred speech|face drooping|face droop|sudden weakness|stroke symptoms)\b",
    r"\b(unconscious|not responding|severe bleeding|bleeding heavily)\b",
]

# Compile the expressions once so they can be reused efficiently.
URGENT_REGEX = re.compile(
    "|".join(URGENT_PATTERNS),
    re.IGNORECASE
)


class LocalLanguageEngine:
    """
    Local natural-language understanding engine for Predict-Care AI.

    Combines predefined phrase matching with a trained intent classifier.
    It also provides access to the existing KTAS prediction pipeline.
    """

    def __init__(self, confidence_threshold=0.40):
        """
        Initialize the engine and train its intent classifier.

        Args:
            confidence_threshold:
                Minimum classification probability required to accept
                a machine-learning prediction when no rule matches.
        """

        self.confidence_threshold = confidence_threshold
        self.intent_model = None

        # Build the intent classifier only when scikit-learn is installed.
        if SKLEARN_AVAILABLE:
            sentences = []
            labels = []

            # Convert the training dictionary into examples and labels.
            for intent, examples in TRAINING_DATA.items():
                sentences.extend(examples)
                labels.extend([intent] * len(examples))

            # TF-IDF converts text into numerical features.
            # Logistic Regression predicts the most likely intent.
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

            # Train the classifier using the predefined examples.
            self.intent_model.fit(sentences, labels)

    @staticmethod
    def normalize(text):
        """
        Normalize user input for consistent phrase matching.

        Converts text to lowercase, removes unsupported punctuation,
        and removes unnecessary whitespace.
        """

        text = str(text).lower()
        text = re.sub(r"[^a-z0-9'\s:/-]", " ", text)

        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def detect_urgent_symptoms(text):
        """
        Check whether a message matches predefined emergency patterns.

        Returns True when a pattern matches and False otherwise.
        """

        return bool(URGENT_REGEX.search(str(text)))

    
    @staticmethod
    def extract_complaint(text):
        """
        Extract a chief complaint from a conversational triage request.

        Removes common commands and patient-related introductory phrases
        while preserving the symptom description for KTAS prediction.
        """

        complaint = str(text).strip()

        if not complaint:
            return ""

        patterns = [
            r"^\s*/(?:triage|predict)\s+",
            r"^\s*predict\s+(?:the\s+)?(?:triage|KTAS)(?:\s+level)?\s+(?:for|of)\s+",
            r"^\s*predict\s+(?:the\s+)?(?:triage|KTAS)\s+",
            r"^\s*(?:run\s+)?(?:a\s+)?triage\s+prediction\s+for\s+",
            r"^\s*classify\s+(?:the\s+)?(?:symptom|complaint)?\s*:?\s*",
            r"^\s*(?:chief complaint|complaint)\s*:\s*",

            # Remove common patient-introduction phrases.
            r"^\s*(?:my|the|a)\s+patient\s+(?:is\s+)?experiencing\s+",
            r"^\s*(?:my|the|a)\s+patient\s+(?:has|have)\s+",
            r"^\s*(?:my|the|a)\s+patient\s+(?:reports|reported)\s+",
            r"^\s*(?:my|the|a)\s+patient\s+(?:complains\s+of|complaining\s+of)\s+",
            r"^\s*(?:my|the|a)\s+patient\s+(?:feels|is\s+feeling)\s+",

            # Remove common pronoun-based introductions.
            r"^\s*(?:he|she|they)\s+(?:is|are)\s+experiencing\s+",
            r"^\s*(?:he|she|they)\s+(?:has|have)\s+",
            r"^\s*(?:he|she|they)\s+(?:reports|report)\s+",

            # Remove first-person symptom introductions.
            r"^\s*i\s+(?:am\s+experiencing|have|feel)\s+",
            r"^\s*i\s+am\s+suffering\s+from\s+",
        ]

        # Repeatedly remove matching prefixes until no further match exists.
        # This also supports requests such as:
        # "/triage my patient is experiencing abdominal discomfort"
        for _ in range(3):
            previous = complaint

            for pattern in patterns:
                updated = re.sub(
                    pattern,
                    "",
                    complaint,
                    count=1,
                    flags=re.IGNORECASE
                )

                if updated != complaint:
                    complaint = updated.strip()
                    break

            if complaint == previous:
                break

        # Remove surrounding punctuation without altering the symptom itself.
        complaint = complaint.strip(" .?!,;:")

        return complaint


    @staticmethod
    def contains_phrase(message, phrases):
        """
        Check whether a message contains any specified phrase.

        Regular-expression boundaries reduce accidental partial-word
        matches when searching for predefined expressions.
        """

        return any(
            re.search(
                r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
                message
            )
            for phrase in phrases
        )

    def classify_intent(self, text):
        """
        Identify the user's conversational intent.

        Recognition follows this order:
        1. Direct commands.
        2. Predefined phrase rules.
        3. Patient and symptom context.
        4. Machine-learning classification.
        5. UNKNOWN when confidence is insufficient.
        """

        message = self.normalize(text)

        if not message:
            return "UNKNOWN"

        # Handle direct commands before general intent classification.
        if message in ("/help", "help"):
            return "HELP"

        if message in ("/quit", "/exit", "quit", "exit"):
            return "GOODBYE"

        # Identify questions about model reliability and clinical safety.
        limitation_phrases = (
            "diagnosis",
            "clinical assessment",
            "clinically accurate",
            "limitations",
            "replace a doctor",
            "replace medical professionals",
            "trust the prediction",
            "difference between prediction and diagnosis",
            "prediction versus diagnosis",
            "prediction vs diagnosis",
            "how reliable",
            "how accurate",
            "can the model be wrong",
            "can the ai be wrong",
            "ai generated results",
            "clinically validated",
            "medical advice",
        )

        if self.contains_phrase(message, limitation_phrases):
            return "EXPLAIN_LIMITATIONS"

        # Recognize medication and pharmacy-related requests.
        medication_phrases = (
            "medication",
            "medicine collection",
            "collect my medicine",
            "pharmacy",
            "chronic medication",
            "collect my pills",
            "pharmacy queue",
            "prescription",
            "dispensary",
            "pharmacist",
            "medicine pickup",
            "tablets",
        )

        if self.contains_phrase(message, medication_phrases):
            return "PREDICT_MEDICATION_WAIT"

        # Recognize doctor waiting time and hospital-duration questions.
        wait_phrases = (
            "waiting time",
            "wait time",
            "how long will i wait",
            "hospital queue",
            "how long is the queue",
            "how long until i see a doctor",
            "if i arrive at",
            "when i arrive at",
            "before my consultation",
            "before a doctor sees",
            "doctor waiting",
            "doctor wait",
            "time in hospital",
            "hospital stay",
            "hospital visit",
            "hospital duration",
            "before i leave the hospital",
            "queue time",
            "consultation delay",
            "physician attend to me",
            "physician sees me",
            "physician see me",
            "medical practitioner see me",
            "be attended to",
            "before i am attended to",
            "before i see the physician",
            "before i see the doctor",
        )

        if self.contains_phrase(message, wait_phrases):
            return "PREDICT_WAIT"

        # Identify requests for explanations of KTAS and triage levels.
        ktas_explanation_phrases = (
            "triage urgency categories",
            "triage urgency levels",
            "five triage levels",
            "five triage categories",
            "triage priority levels",
            "patient acuity levels",
            "hospital triage categories",
            "triage scale",
            "korean triage and acuity scale",
        )

        explanation_starters = (
            "explain",
            "what are",
            "what is",
            "describe",
            "tell me about",
            "how many",
            "what does",
            "could you explain",
        )

        if (
            self.contains_phrase(message, ktas_explanation_phrases)
            or (
                self.contains_phrase(message, explanation_starters)
                and self.contains_phrase(
                    message,
                    (
                        "ktas",
                        "triage levels",
                        "triage categories",
                        "acuity",
                        "triage urgency",
                        "triage priority",
                    )
                )
            )
        ):
            return "EXPLAIN_KTAS"

        # Recognize explicit requests to predict or classify triage.
        prediction_phrases = (
            "predict triage",
            "predict ktas",
            "classify",
            "triage for",
            "ktas for",
            "chief complaint",
            "complaint:",
            "triage prediction",
            "assess the urgency",
            "triage assessment",
            "ktas category for",
        )

        if self.contains_phrase(message, prediction_phrases):
            return "PREDICT_TRIAGE"

        # Patient-context phrases help identify symptom descriptions.
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
            "experiencing",
            "reports",
            "feels",
        )

        # Symptom vocabulary used to identify potential triage requests.
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
            "abdominal discomfort",
            "stomach discomfort",
            "stomach cramps",
            "abdominal pain",
            "back pain",
            "chest discomfort",
        )

        has_patient_context = self.contains_phrase(
            message,
            patient_context
        )

        has_symptom = self.contains_phrase(
            message,
            symptom_terms
        )

        if has_patient_context and has_symptom:
            return "PREDICT_TRIAGE"

        # General KTAS questions are treated as explanation requests.
        if "ktas" in message or "acuity scale" in message:
            return "EXPLAIN_KTAS"

        # Recognize common greetings, including some local expressions.
        greeting_phrases = (
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "good day",
            "greetings",
            "howzit",
            "sawubona",
            "dumelang",
        )

        if self.contains_phrase(message, greeting_phrases):
            return "GREETING"

        # Recognize questions about available system functionality.
        help_phrases = (
            "what can you do",
            "what can i ask",
            "show commands",
            "how do i use this",
            "what services do you offer",
            "show available features",
            "show help menu",
            "list your commands",
            "what questions can i ask",
            "what can this system do",
        )

        if self.contains_phrase(message, help_phrases):
            return "HELP"

        # Recognize symptom-only messages as potential triage requests.
        if self.contains_phrase(message, symptom_terms):
            return "PREDICT_TRIAGE"

        # Use the trained classifier when predefined rules do not match.
        if self.intent_model is not None:
            probabilities = self.intent_model.predict_proba(
                [message]
            )[0]

            best_index = probabilities.argmax()
            confidence = float(probabilities[best_index])

            if confidence >= self.confidence_threshold:
                return str(
                    self.intent_model.classes_[best_index]
                )

        # Reject requests that cannot be classified confidently.
        return "UNKNOWN"

    def run_triage_prediction(self, complaint):
        """
        Send a chief complaint to the existing KTAS prediction pipeline.

        Critical symptom expressions are checked before prediction.
        The returned KTAS result is an experimental model output.
        """

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

        # Prevent the model from providing routine triage predictions
        # for selected descriptions of potentially critical symptoms.
        if self.contains_phrase(
            normalized_complaint,
            critical_phrases
        ):
            return (
                "SAFETY ALERT: This description may indicate a "
                "medical emergency. Contact emergency services "
                "immediately. In South Africa, call 112 from a "
                "mobile phone or 10177 for an ambulance. Do not "
                "wait for a Predict-Care AI prediction."
            )

        if not complaint:
            return (
                "Please provide a chief complaint. "
                "Example: /triage headache"
            )

        # Explain the problem if the existing pipeline is unavailable.
        if predict_ktas is None:
            return (
                "The unified prediction pipeline could not be loaded.\n"
                "Check that you are running this program from the "
                "project root and that the model files and "
                "dependencies are present.\n"
                f"Import detail: {TRIAGE_IMPORT_ERROR}"
            )

        try:
            # Obtain the prediction from the existing KTAS model.
            result = predict_ktas(complaint)

            return (
                f"Existing NLP classifier output for '{complaint}': "
                f"KTAS {result['predicted_ktas']} "
                f"({result['emergency_category']}).\n\n"
                "This is an experimental model output, not a "
                "clinical assessment. It cannot guarantee that "
                "a complaint is non-urgent. Seek professional "
                "help for concerning symptoms."
            )

        except Exception as error:
            # Return a clear error instead of inventing a prediction.
            return (
                "The existing NLP prediction failed. "
                "No prediction was invented.\n"
                f"Error: {type(error).__name__}: {error}"
            )

    def respond(self, text):
        """
        Process a user message and generate the appropriate response.

        Returns:
            tuple: (response_text, should_exit)
        """

        message = self.normalize(text)

        # Safety checks take priority over ordinary chatbot responses.
        if self.detect_urgent_symptoms(text):
            return URGENT_RESPONSE, False

        # Handle commands for ending, resetting, or requesting help.
        if message in ("/quit", "/exit", "quit", "exit"):
            return RESPONSES["GOODBYE"], True

        if message == "/help":
            return RESPONSES["HELP"], False

        if message == "/reset":
            return (
                "Ready for a new request. "
                "What would you like to do?",
                False
            )

        # Direct triage commands bypass the general intent classifier.
        if (
            message.startswith("/triage ")
            or message.startswith("/predict ")
        ):
            complaint = self.extract_complaint(text)
            return self.run_triage_prediction(complaint), False

        # Identify the user's intent using rules and the local classifier.
        intent = self.classify_intent(text)

        if intent == "PREDICT_TRIAGE":
            complaint = self.extract_complaint(text)

            # Ask for a complaint when no meaningful symptom is supplied.
            if not complaint or complaint.lower() in (
                "this complaint",
                "the complaint",
                "triage",
                "ktas",
            ):
                return (
                    "I recognized a triage-prediction request. "
                    "Please include the complaint, for example: "
                    "/triage headache",
                    False
                )

            return self.run_triage_prediction(complaint), False

        # Return the predefined response for recognized non-triage intents.
        if intent in RESPONSES:
            return RESPONSES[intent], False

        # Provide guidance when no supported intent is recognized.
        return (
            "I could not confidently identify that request. "
            "Try /help, ask 'What is KTAS?', or enter /triage "
            "followed by a chief complaint.",
            False
        )


def main():
    """
    Run the interactive command-line version of the language engine.
    """

    # Initialize the language engine and train its local intent classifier.
    assistant = LocalLanguageEngine()

    print("=" * 55)
    print("Predict-Care AI — Local Language Engine")
    print("No Ollama or cloud language model is required.")

    # Display whether the machine-learning classifier is available.
    print(
        "Intent classifier:",
        "available" if assistant.intent_model is not None
        else "rules only (scikit-learn unavailable)"
    )

    # Display whether the existing KTAS prediction pipeline was loaded.
    print(
        "Unified prediction pipeline:",
        "available" if TRIAGE_MODEL_AVAILABLE else "not loaded"
    )

    print("Type /help for commands or /quit to exit.")
    print("=" * 55)

    # Continue accepting user messages until the user exits.
    while True:
        try:
            user_text = input("\nYou: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Goodbye.")
            break

        if not user_text:
            print("Assistant: Please enter a message.")
            continue

        # Generate the response and determine whether to end the session.
        response, should_exit = assistant.respond(user_text)

        print(f"\nAssistant: {response}")

        if should_exit:
            break


# Start the terminal application only when this file is executed directly.
if __name__ == "__main__":
    main()
