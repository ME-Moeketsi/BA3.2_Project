"""
Predict-Care AI — Local Language Understanding Engine

Here I am making the local chat front-end able to handle four kinds
of requests end to end:

    GREETING / HELP / EXPLAIN_*   -> fixed responses
    PREDICT_TRIAGE                -> pipeline.route(complaint)
    PREDICT_WAIT                  -> pipeline.route(request, patient_info)
    PREDICT_MEDICATION_WAIT       -> explained, not predicted (no model)

Behaviours:

  * 'high risk', 'urgent', 'critical' etc. are treated as priority
    hints, so a user does not have to remember the numeric scale.
  * A user-stated arrival time ('8am', '08:00', 'at 20h00') is
    respected and mapped into the 0–240-minute simulated window
    by clamping, NOT by wrapping modulo (which would collapse
    8am, 12pm, 4pm and 8pm all to 0). If none is given, the system
    clock is used.
  * When the user does not mention doctor count or peak status,
    the engine falls back to the most common values in the training
    data and says so explicitly in the reply.
  * Follow-up turns ("just tell me how long i will wait",
    "what about the total time?") are recognised as waiting-time
    requests when the previous turn already has patient context.

If a waiting-time question also contains a clinical description
('fever for five days, headache'), the engine runs KTAS first and
maps the KTAS level to a waiting-time priority, so the user gets a
triage answer *and* a waiting-time answer from one question.

Replies are phrased in natural language: the engine refers to
'your description' rather than quoting the whole sentence, and uses
prose rather than '=' / 'key:value' formatting.

No Ollama, no cloud API. All prediction goes through pipeline.route().
"""

import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from pipeline import route
    PIPELINE_AVAILABLE = True
    PIPELINE_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001
    route = None
    PIPELINE_AVAILABLE = False
    PIPELINE_IMPORT_ERROR = str(error)


# ============================================================
# TRAINING DATA FOR THE INTENT CLASSIFIER
# ============================================================

TRAINING_DATA = {
    "GREETING": [
        "hello", "hi", "hey", "good morning",
        "good afternoon", "good evening", "how are you",
    ],
    "HELP": [
        "help", "what can you do", "what can I ask",
        "show me your commands", "how do I use this system",
    ],
    "EXPLAIN_KTAS": [
        "what is KTAS", "explain KTAS",
        "what does KTAS stand for",
        "explain the Korean Triage and Acuity Scale",
        "explain triage in simple language",
    ],
    "PREDICT_TRIAGE": [
        "predict triage for headache",
        "predict KTAS for fever",
        "classify this chief complaint",
        "run a triage prediction for chest pain",
        "what does the model predict for dizziness",
        "classify the symptom nausea",
        "predict the triage level for abdominal pain",
    ],
    "PREDICT_WAIT": [
        "how long will I wait",
        "predict my waiting time",
        "how long until I see a doctor",
        "what is the hospital waiting time",
        "how long is the queue",
        "estimate my waiting time at the hospital",
        "just tell me how long I will wait",
        "what about the total time",
    ],
    "PREDICT_MEDICATION_WAIT": [
        "how long will I wait for medicine",
        "I am collecting chronic medication",
        "predict pharmacy waiting time",
        "how long until I collect my medication",
        "how long will I wait to collect my pills",
    ],
    "EXPLAIN_LIMITATIONS": [
        "is this a medical diagnosis",
        "can this replace a doctor",
        "what are the model limitations",
        "is this a clinical assessment",
        "what is the difference between prediction and diagnosis",
    ],
    "GOODBYE": [
        "goodbye", "bye", "see you later", "exit", "quit",
    ],
}


RESPONSES = {
    "GREETING": (
        "Hello! I am the Predict-Care AI local assistant. "
        "You can ask me about KTAS, describe a patient for an "
        "experimental triage prediction, or ask a waiting-time "
        "question. For example: 'a patient with chest pain', or "
        "'high risk, 2 doctors, peak, how long will I wait?'."
    ),
    "HELP": (
        "Here is what I can do:\n"
        "1. Explain KTAS.\n"
        "2. Predict a KTAS level from a chief complaint.\n"
        "3. Predict doctor wait and total in-hospital time. "
        "You can give priority as a number or as 'low', 'medium' or "
        "'high risk', state an arrival time, and mention whether it "
        "is a peak period. If you leave out the doctor count or peak "
        "status, I will assume the most common value in the training "
        "data and tell you I did.\n"
        "4. If your question contains a clinical description, I will "
        "run KTAS first and use that level for the waiting-time "
        "prediction.\n"
        "5. Explain the limitations of AI predictions.\n\n"
        "Type /quit to exit."
    ),
    "EXPLAIN_KTAS": (
        "KTAS stands for Korean Triage and Acuity Scale. "
        "It is a system used to prioritise patients according to "
        "urgency. A Predict-Care AI output is an experimental model "
        "prediction, not a clinical assessment."
    ),
    "EXPLAIN_LIMITATIONS": (
        "A prediction is an estimate produced by a machine-learning "
        "model using patterns learned from training data. A clinical "
        "assessment is performed by a qualified healthcare "
        "professional. A model prediction is not a diagnosis and must "
        "not replace professional assessment. Predict-Care AI is an "
        "academic prototype."
    ),
    "PREDICT_MEDICATION_WAIT": (
        "I recognised your medication-collection request. The "
        "medication-wait model is not part of the current system, so "
        "I will not invent a value. If you meant the doctor waiting "
        "time, ask e.g. 'priority 2, 3 doctors, not peak, how long "
        "will I wait for a doctor?'."
    ),
    "GOODBYE": (
        "Goodbye! Remember that Predict-Care AI is an academic "
        "prototype, not a medical service."
    ),
}


# ============================================================
# URGENT SAFETY CHECK
# ============================================================

URGENT_PATTERNS = [
    r"\bchest pain\b.{0,100}\b(difficulty breathing|can't breathe|cannot breathe|"
    r"shortness of breath|struggling to breathe)\b",
    r"\b(difficulty breathing|can't breathe|cannot breathe|"
    r"struggling to breathe|severe breathing difficulty)\b",
    r"\b(slurred speech|face drooping|face droop|sudden weakness|"
    r"stroke symptoms)\b",
    r"\b(unconscious|not responding|severe bleeding|bleeding heavily)\b",
]

URGENT_REGEX = re.compile("|".join(URGENT_PATTERNS), re.IGNORECASE)

URGENT_RESPONSE = (
    "Please treat this as urgent. Your message mentions symptoms that "
    "may require emergency attention — chest pain with difficulty "
    "breathing, severe breathing difficulty, signs of stroke, "
    "unconsciousness, or severe bleeding. Seek emergency medical help "
    "now. In South Africa, call 112 from a mobile phone or 10177 for "
    "an ambulance. Do not wait for this prototype to assess you."
)


# ============================================================
# SLOT EXTRACTION FOR WAITING-TIME QUESTIONS
# ============================================================

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}

_PRIORITY_HIGH_HINTS = re.compile(
    r"\b(high[- ]?risk|high[- ]?priority|urgent|critical|emergency|"
    r"serious|severe|very ill)\b",
    re.IGNORECASE,
)
_PRIORITY_MEDIUM_HINTS = re.compile(
    r"\b(medium[- ]?risk|medium[- ]?priority|moderate[- ]?risk|"
    r"moderate[- ]?priority|semi[- ]?urgent)\b",
    re.IGNORECASE,
)
_PRIORITY_LOW_HINTS = re.compile(
    r"\b(low[- ]?risk|low[- ]?priority|routine|non[- ]?urgent|"
    r"mild|minor)\b",
    re.IGNORECASE,
)


def _find_priority(text):
    match = re.search(r"\bpriority(?:\s+level)?\s*[:=]?\s*([123])\b", text)
    if match:
        return int(match.group(1))
    for word, value in _NUMBER_WORDS.items():
        if value in (1, 2, 3) and re.search(
            r"\bpriority\s+" + word + r"\b", text
        ):
            return value
    if _PRIORITY_HIGH_HINTS.search(text):
        return 3
    if _PRIORITY_MEDIUM_HINTS.search(text):
        return 2
    if _PRIORITY_LOW_HINTS.search(text):
        return 1
    return None


def _find_num_doctors(text):
    match = re.search(r"\b(\d+)\s+doctors?\b", text)
    if match:
        return int(match.group(1))
    for word, value in _NUMBER_WORDS.items():
        if re.search(r"\b" + word + r"\s+doctors?\b", text):
            return value
    match = re.search(r"\bdoctors?\s*[:=]\s*(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _find_peak(text):
    if re.search(
        r"\b(not\s+peak|non[- ]?peak|off[- ]?peak|normal hours|quiet)\b",
        text,
    ):
        return 0
    if re.search(r"\b(peak|busy|rush|high demand|peak arrival)\b", text):
        return 1
    return None


def _find_arrival_time(text):
    """
    Here I am converting a wall-clock time into the 0–240-minute
    simulated window.

    Important: the window is an OFFSET, not an hour of the day, so
    we CLAMP into [0, 239] rather than taking modulo 240. Taking
    modulo would collapse 8am, 12pm, 4pm and 8pm all to 0 — which
    is exactly the bug we are fixing here.
    """
    lowered = text.lower()
    named_block = re.search(
        r"\b(?:tomorrow\s+)?(morning|afternoon|evening|night)\b", lowered
    )

    explicit = None
    # 24-hour with colon: 08:00, 20:30
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lowered)
    if match:
        explicit = int(match.group(1)) * 60 + int(match.group(2))
    # 24-hour with 'h': 20h00, 8h30
    if explicit is None:
        match = re.search(r"\b([01]?\d|2[0-3])h([0-5]\d)\b", lowered)
        if match:
            explicit = int(match.group(1)) * 60 + int(match.group(2))
    # 12-hour with am/pm: 8am, 8:30 pm, 12 pm
    if explicit is None:
        match = re.search(
            r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)\b", lowered
        )
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            if match.group(3) == "pm" and hour != 12:
                hour += 12
            if match.group(3) == "am" and hour == 12:
                hour = 0
            explicit = hour * 60 + minute
    # Bare hour after 'at': 'at 8', 'at 20'
    if explicit is None:
        match = re.search(r"\bat\s+([01]?\d|2[0-3])\b", lowered)
        if match:
            explicit = int(match.group(1)) * 60

    if explicit is not None:
        return max(0, min(explicit, 239))

    if named_block:
        hour_map = {"morning": 8, "afternoon": 14, "evening": 19, "night": 22}
        return min(hour_map[named_block.group(1)] * 60, 239)

    return None


def _find_intent_for_wait(text):
    if re.search(r"\b(both|all)\b", text):
        return "both"
    doctor = re.search(
        r"\b(doctor|physician|consultation|see a doctor|see the doctor)\b",
        text,
    )
    total = re.search(
        r"\b(total|overall|in hospital|length of stay|"
        r"leave the hospital|discharged?)\b",
        text,
    )
    if doctor and total:
        return "both"
    if doctor:
        return "doctor"
    if total:
        return "total"
    if re.search(r"\bwait(ing)?\b", text):
        return "both"
    return None


def _looks_like_clinical_description(text):
    symptom_terms = (
        "fever", "headache", "pain", "chest", "abdominal", "vomiting",
        "nausea", "dizziness", "dizzy", "bleeding", "wound", "burn",
        "swelling", "injury", "cough", "breathing", "shortness of breath",
        "unconscious", "seizure", "rash", "sore throat", "diarrhoea",
        "diarrhea", "fatigue", "weakness", "unresponsive",
    )
    duration_patterns = (
        r"\b\d+\s*(day|days|hour|hours|week|weeks|month|months)\b",
        r"\b(since yesterday|since last night|for the last|"
        r"for \d+ (day|days|hour|hours|week|weeks))\b",
    )
    return any(term in text for term in symptom_terms) or any(
        re.search(p, text) for p in duration_patterns
    )


def _ktas_to_wait_priority(ktas_level):
    if ktas_level in (1, 2, 3):
        return 3
    if ktas_level == 4:
        return 2
    return 1


def _join_naturally(items):
    """Here I am turning a list into 'a, b and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return items[0] + " and " + items[1]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ============================================================
# ENGINE
# ============================================================

class LocalLanguageEngine:

    def __init__(self, confidence_threshold=0.40):
        self.confidence_threshold = confidence_threshold
        self.intent_model = None
        self.last_patient_info = None
        self.last_prediction_target = None

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
                        strip_accents="unicode",
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
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

    # ------------------------------------------------------------------
    # Triage path
    # ------------------------------------------------------------------

    def _run_triage(self, complaint):
        if route is None:
            return (
                "The prediction pipeline is not available right now. "
                f"Technical detail: {PIPELINE_IMPORT_ERROR}"
            )

        if not complaint:
            return (
                "I need a chief complaint to run the triage classifier. "
                "For example: 'a patient with chest pain'."
            )

        result = route(complaint)

        if result.get("type") == "error":
            return (
                "I could not run the triage classifier on that. "
                f"Technical detail: {result['message']}"
            )
        if result.get("type") != "ktas":
            return "Something went wrong inside the prediction pipeline."

        r = result["result"]
        return (
            f"Based on your description, the triage classifier "
            f"estimates KTAS level {r['predicted_ktas']} "
            f"({r['emergency_category'].lower()}). "
            "This is a model output, not a clinical assessment."
        )

    # ------------------------------------------------------------------
    # Waiting-time path
    # ------------------------------------------------------------------

    def _run_wait(self, text):
        if route is None:
            return (
                "The prediction pipeline is not available right now. "
                f"Technical detail: {PIPELINE_IMPORT_ERROR}"
            )

        normalized = self.normalize(text)

        priority = _find_priority(normalized)
        num_doctors = _find_num_doctors(normalized)
        is_peak = _find_peak(normalized)
        stated_arrival = _find_arrival_time(normalized)

        prediction_type = _find_intent_for_wait(normalized) or "both"

        # Here I am running KTAS first when the sentence contains a
        # clinical description AND the user did not state a priority.
        triage_sentence = ""
        if priority is None and _looks_like_clinical_description(normalized):
            complaint = self._extract_complaint(text)
            triage_result = route(complaint)
            if triage_result.get("type") == "ktas":
                r = triage_result["result"]
                priority = _ktas_to_wait_priority(r["predicted_ktas"])
                triage_sentence = (
                    f"Based on your description, the triage classifier "
                    f"estimates KTAS level {r['predicted_ktas']} "
                    f"({r['emergency_category'].lower()}). "
                    f"I will use that to set the waiting-time priority "
                    f"to {priority}, which means "
                    f"{self._priority_label(priority)}. "
                )

        # Here I am falling back to the previous turn for any slot the
        # user did not restate.
        if self.last_patient_info:
            if priority is None:
                priority = self.last_patient_info.get("Priority")
            if is_peak is None:
                is_peak = self.last_patient_info.get("IsPeakArrival")
            if num_doctors is None:
                num_doctors = self.last_patient_info.get("NumDoctors")
            if stated_arrival is None and "arrival_time" not in normalized:
                # Carry the previous arrival forward only if the user
                # did not state a new one this turn.
                stated_arrival = self.last_patient_info.get("ArrivalTime")

        # Here I am reading the training-data defaults from the bundle
        # so any silent assumption is derived, not invented.
        defaults = self._load_defaults()
        assumed = []
        if num_doctors is None:
            num_doctors = defaults.get("NumDoctors")
            if num_doctors is not None:
                assumed.append(
                    f"the number of doctors on shift as {num_doctors}, "
                    "the most common staffing level in the training data"
                )
        if is_peak is None:
            is_peak = defaults.get("IsPeakArrival")
            if is_peak is not None:
                assumed.append(
                    "the arrival period as "
                    + ("peak" if is_peak == 1 else "not peak")
                    + ", the most common value in the training data"
                )

        # Here I am collecting only what is still missing.
        missing = []
        if priority is None:
            missing.append(
                "the priority level (low, medium or high risk)"
            )
        if num_doctors is None:
            missing.append("how many doctors are on shift")
        if is_peak is None:
            missing.append(
                "whether you are arriving during a peak period"
            )

        if missing:
            if triage_sentence:
                opener = triage_sentence
            else:
                opener = "I can run the waiting-time model, but "
            return (
                opener
                + "I still need to know "
                + _join_naturally(missing)
                + ". You can include them in one sentence, for example: "
                "'high risk, 2 doctors, peak, how long will I wait?'."
            )

        # Here I am choosing the arrival time: user-stated first,
        # system clock otherwise.
        if stated_arrival is not None:
            arrival_time = stated_arrival
            arrival_source = "the time you gave"
        else:
            from main_menu import get_arrival_time_from_system_clock
            arrival_time = get_arrival_time_from_system_clock()
            arrival_source = "the current time"

        patient_info = {
            "Priority": int(priority),
            "ArrivalTime": int(arrival_time),
            "NumDoctors": int(num_doctors),
            "IsPeakArrival": int(is_peak),
        }
        self.last_patient_info = patient_info
        self.last_prediction_target = prediction_type

        result = route(text, patient_info=patient_info)

        if result.get("type") == "error":
            return triage_sentence + (
                "I could not run the waiting-time model. "
                f"Technical detail: {result['message']}"
            )
        if result.get("type") != "waiting_time":
            return triage_sentence + "Something went wrong inside the pipeline."

        message = result.get("message", "")
        doctor_minutes = result.get("doctor_minutes")
        total_minutes = result.get("total_minutes")

        # Here I am composing the reply in prose, in this order:
        #   1. triage sentence (if any)
        #   2. the waiting-time answer
        #   3. numeric estimates
        #   4. what was used for the estimate
        #   5. stated assumptions (as separate sentences)
        #   6. safety line
        parts = []
        if triage_sentence:
            parts.append(triage_sentence.strip())
        parts.append(message.strip())

        if doctor_minutes is not None and total_minutes is not None:
            parts.append(
                f"Concretely, that is about {doctor_minutes:.0f} minutes "
                f"until a doctor is seen and about "
                f"{total_minutes:.0f} minutes in the hospital in total."
            )

        context_bits = [
            f"priority {patient_info['Priority']} "
            f"({self._priority_label(patient_info['Priority'])})",
            f"{patient_info['NumDoctors']} doctor"
            f"{'s' if patient_info['NumDoctors'] != 1 else ''}",
            "peak arrival" if patient_info["IsPeakArrival"] == 1
            else "non-peak arrival",
            f"arrival around minute {patient_info['ArrivalTime']} "
            f"of the simulated window ({arrival_source})",
        ]
        parts.append(
            "For reference, I used "
            + _join_naturally(context_bits)
            + "."
        )

        if assumed:
            assumption_sentences = ["I assumed " + item + "." for item in assumed]
            parts.append(
                " ".join(assumption_sentences)
                + " Please tell me if that is wrong and I will redo "
                  "the estimate."
            )

        parts.append(
            "This is a simulation-based estimate, not a clinical "
            "assessment."
        )

        return " ".join(parts)

    @staticmethod
    def _priority_label(priority):
        return {
            1: "low risk",
            2: "medium risk",
            3: "high risk",
        }.get(int(priority), "unknown")

    @staticmethod
    def _load_defaults():
        """Here I am reading the training-data defaults from the model."""
        try:
            import joblib
            from pipeline import WAIT_MODEL_PATH
            bundle = joblib.load(WAIT_MODEL_PATH)
            return bundle.get("defaults", {}) if isinstance(bundle, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    # Intent routing
    # ------------------------------------------------------------------

    def classify_intent(self, text):
        message = self.normalize(text)
        if not message:
            return "UNKNOWN"

        if message in ("/help", "help"):
            return "HELP"
        if message in ("/quit", "/exit", "quit", "exit"):
            return "GOODBYE"
        if message == "/reset":
            return "RESET"

        explanation_phrases = (
            "diagnosis", "clinical assessment", "limitations",
            "replace a doctor",
            "difference between prediction and diagnosis",
            "prediction versus diagnosis", "prediction vs diagnosis",
        )
        if any(phrase in message for phrase in explanation_phrases):
            return "EXPLAIN_LIMITATIONS"

        medication_phrases = (
            "medication", "medicine collection", "collect my medicine",
            "pharmacy", "chronic medication", "collect my pills",
        )
        if any(phrase in message for phrase in medication_phrases):
            return "PREDICT_MEDICATION_WAIT"

        wait_phrases = (
            "waiting time", "wait time", "how long will i wait",
            "how long until i see a doctor",
            "how long will i wait for a doctor",
            "how long will i wait for the doctor",
            "how long is the queue", "queue",
            "total time in hospital", "length of stay",
            "time in the hospital", "in hospital", "doctor wait",
            "wait in a queue",
        )
        if any(phrase in message for phrase in wait_phrases):
            return "PREDICT_WAIT"

        if (
            re.search(r"\b(predict|estimate|calculate)\b", message)
            and re.search(r"\b(priority|doctors?|peak)\b", message)
        ):
            return "PREDICT_WAIT"

        if re.search(r"\bpriority\s*(level)?\s*[:=]?\s*[123]\b", message):
            return "PREDICT_WAIT"
        if re.search(r"\b\d+\s+doctors?\b", message):
            return "PREDICT_WAIT"
        if (
            re.search(r"\b(peak|non[- ]?peak)\b", message)
            and "doctor" in message
        ):
            return "PREDICT_WAIT"

        # Here I am treating a bare follow-up ("just tell me how long
        # i will wait", "what about the total time?") as a waiting-time
        # request when the previous turn already has patient context.
        if self.last_patient_info is not None:
            if re.search(
                r"\b(how long|what about|and the|just tell me|"
                r"wait|waiting|total|overall)\b",
                message,
            ):
                return "PREDICT_WAIT"

        prediction_phrases = (
            "predict", "classify", "triage for", "ktas for",
            "chief complaint", "complaint:",
        )
        if any(phrase in message for phrase in prediction_phrases):
            return "PREDICT_TRIAGE"

        patient_context = (
            "patient", "he has", "she has", "they have",
            "he is", "she is", "they are",
            "complaining of", "suffering from",
            "says he", "says she",
        )
        symptom_terms = (
            "wound", "fever", "pain", "pains",
            "chest pain", "chest pains",
            "bleeding", "burn", "swelling", "injury",
            "dizzy", "dizziness", "vomiting", "nausea",
            "breathing", "unconscious", "unresponsive",
            "headache", "shortness of breath",
            "heart attack", "cardiac arrest", "stroke", "seizure",
        )
        has_patient_context = any(p in message for p in patient_context)
        has_symptom = any(s in message for s in symptom_terms)
        if has_patient_context and has_symptom:
            return "PREDICT_TRIAGE"

        if "ktas" in message or "acuity scale" in message:
            return "EXPLAIN_KTAS"

        if (
            message in ("hello", "hi", "hey")
            or message.startswith((
                "hello ", "hi ", "hey ",
                "good morning", "good afternoon", "good evening",
            ))
        ):
            return "GREETING"

        if any(phrase in message for phrase in (
            "what can you do", "what can i ask",
            "show commands", "how do i use this",
        )):
            return "HELP"

        if any(s in message for s in symptom_terms):
            return "PREDICT_TRIAGE"

        if self.intent_model is not None:
            probabilities = self.intent_model.predict_proba([message])[0]
            best_index = probabilities.argmax()
            confidence = float(probabilities[best_index])
            if confidence >= self.confidence_threshold:
                return str(self.intent_model.classes_[best_index])

        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Public responder
    # ------------------------------------------------------------------

    def respond(self, text):
        if not text or not text.strip():
            return "Please type a message first.", False

        if self.detect_urgent_symptoms(text):
            return URGENT_RESPONSE, False

        intent = self.classify_intent(text)

        if intent == "GOODBYE":
            return RESPONSES["GOODBYE"], True
        if intent == "HELP":
            return RESPONSES["HELP"], False
        if intent == "RESET":
            self.last_patient_info = None
            self.last_prediction_target = None
            return "Conversation state cleared. What would you like to do?", False
        if intent == "GREETING":
            return RESPONSES["GREETING"], False
        if intent == "EXPLAIN_KTAS":
            return RESPONSES["EXPLAIN_KTAS"], False
        if intent == "EXPLAIN_LIMITATIONS":
            return RESPONSES["EXPLAIN_LIMITATIONS"], False
        if intent == "PREDICT_MEDICATION_WAIT":
            return RESPONSES["PREDICT_MEDICATION_WAIT"], False

        if intent == "PREDICT_TRIAGE":
            complaint = self._extract_complaint(text)
            return self._run_triage(complaint), False

        if intent == "PREDICT_WAIT":
            return self._run_wait(text), False

        return (
            "I could not confidently classify that request. "
            "Try /help, ask 'What is KTAS?', describe a patient "
            "('patient with chest pain'), or give waiting-time "
            "context ('priority 2, 3 doctors, peak, how long will "
            "I wait?')."
        ), False

    @staticmethod
    def _extract_complaint(text):
        complaint = str(text).strip()
        patterns = [
            r"^\s*/(?:triage|predict)\s+",
            r"^\s*predict\s+(?:the\s+)?(?:triage|KTAS)"
            r"(?:\s+level)?\s+(?:for|of)\s+",
            r"^\s*predict\s+(?:the\s+)?(?:triage|KTAS)\s+",
            r"^\s*classify\s+(?:the\s+)?(?:symptom|complaint)?\s*:?\s*",
            r"^\s*(?:chief complaint|complaint)\s*:\s*",
        ]
        for pattern in patterns:
            updated = re.sub(pattern, "", complaint, flags=re.IGNORECASE)
            if updated != complaint:
                return updated.strip(" .?!")
        return complaint.strip(" .?!")


# ============================================================
# CLI
# ============================================================

def main():
    assistant = LocalLanguageEngine()

    print("=" * 60)
    print("Predict-Care AI — Local Language Engine")
    print("No Ollama or cloud model required.")
    print("Intent classifier:",
          "available" if assistant.intent_model is not None
          else "rules only")
    print("Unified pipeline:",
          "available" if PIPELINE_AVAILABLE else "NOT loaded")
    print("Type /help for commands or /quit to exit.")
    print("=" * 60)

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