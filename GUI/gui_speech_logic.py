"""
gui_speech_logic.py
Speech recognition and dictation parsing for Predict-Care AI's Tkinter GUI.
No Tkinter code and no model/prediction code live in this module.

Google Speech Recognition requires an internet connection.
Install: python -m pip install SpeechRecognition PyAudio
"""
import re
import threading

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    sr = None
    SR_AVAILABLE = False

_STATE = {"listening": False, "stop_event": None, "thread": None}
_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100,
}


def _number(text):
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        value = float(match.group(1))
        return int(value) if value.is_integer() else value
    total, found = 0, False
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _NUMBERS:
            found = True
            total = 100 if _NUMBERS[word] == 100 else total + _NUMBERS[word]
    return total if found else None


def _number_after(text, anchors, low, high):
    text = text.lower()
    for anchor in anchors:
        index = text.find(anchor)
        if index >= 0:
            value = _number(text[index + len(anchor):index + len(anchor) + 60])
            if value is not None and low <= value <= high:
                return value
    return None


def parse_dictation(transcript, available_options=None):
    """Parse recognised speech into updates for the GUI's form fields."""
    result = {}
    text = (transcript or "").strip()
    if not text:
        return result
    low = text.lower()

    if re.search(r"\b(male|man|boy)\b", low) and not re.search(r"\bfemale\b", low):
        result["Sex"] = "1"
    elif re.search(r"\b(female|woman|girl)\b", low):
        result["Sex"] = "2"

    age = _number_after(low, ["age", "years old", "year old", "aged"], 0, 120)
    if age is not None:
        result["Age"] = str(int(age))

    bp = re.search(r"blood pressure\s+(?:is\s+)?(\d{2,3})\s*(?:over|on|by|/)\s*(\d{2,3})", low)
    sbp = _number_after(low, ["systolic", "sbp", "blood pressure"], 40, 300)
    dbp = _number_after(low, ["diastolic", "dbp"], 20, 200)
    if bp:
        sbp, dbp = int(bp.group(1)), int(bp.group(2))
    if sbp is not None:
        result["SBP"] = str(int(sbp))
    if dbp is not None:
        result["DBP"] = str(int(dbp))

    for field, anchors, lo, hi, cast in [
        ("HR", ["heart rate", "pulse", "hr is", "hr of"], 20, 250, int),
        ("RR", ["respiratory rate", "resp rate", "breaths per minute", "rr is"], 4, 80, int),
        ("BT", ["body temperature", "temperature", "temp"], 25, 45, float),
        ("Saturation", ["oxygen saturation", "oxygen sat", "saturation", "spo2"], 0, 100, int),
        ("NRS_pain", ["pain score", "pain level", "nrs", "pain is"], 0, 10, float),
    ]:
        value = _number_after(low, anchors, lo, hi)
        if value is not None:
            result[field] = str(cast(value))

    if re.search(r"\b(no injury|not injured|no visible injury)\b", low):
        result["Injury"] = "1 = No"
    elif re.search(r"\b(has injury|injured|visible injury|there is injury)\b", low):
        result["Injury"] = "2 = Yes"

    if re.search(r"\b(not in pain|no pain|pain free)\b", low):
        result["Pain"] = "0"
    elif re.search(r"\b(in pain|has pain|is in pain)\b", low):
        result["Pain"] = "1"

    if re.search(r"\b(unresponsive|no response)\b", low):
        result["Mental"] = "4 = Unresponsive"
    elif re.search(r"\b(pain response|responds to pain)\b", low):
        result["Mental"] = "3 = Pain Response"
    elif re.search(r"\b(verbal response|responds verbally)\b", low):
        result["Mental"] = "2 = Verbal Response"
    elif re.search(r"\b(alert|oriented|awake)\b", low):
        result["Mental"] = "1 = Alert"

    # Test private ambulance before generic ambulance.
    if re.search(r"\bprivate ambulance\b", low):
        result["Arrival mode"] = "4 = Private Ambulance"
    elif re.search(r"\b(public ambulance|ambulance)\b", low):
        result["Arrival mode"] = "2 = Public Ambulance"
    elif re.search(r"\b(walking|walked in|ambulatory)\b", low):
        result["Arrival mode"] = "1 = Walking"
    elif re.search(r"\b(private vehicle|car|taxi)\b", low):
        result["Arrival mode"] = "3 = Private Vehicle"
    elif re.search(r"\bwheelchair\b", low):
        result["Arrival mode"] = "5 = Wheelchair"
    elif re.search(r"\b(stretcher|gurney)\b", low):
        result["Arrival mode"] = "6 = Stretcher"

    markers = ("chief complaint", "complaining of", "complains of",
               "presenting with", "came in with", "the patient has")
    complaint = None
    for marker in markers:
        index = low.find(marker)
        if index >= 0:
            complaint = text[index + len(marker):].strip(" .,")
            break
    if complaint is None and len(text) > 5:
        complaint = text.strip(" .,")
    if complaint:
        result["Chief_complain"] = complaint
    return result


def speech_available():
    if not SR_AVAILABLE:
        return False
    try:
        return bool(sr.Microphone.list_microphone_names())
    except Exception:
        return False


def start_dictation(on_transcript, on_status, on_error):
    """Record until stop_dictation() is called, then transcribe all captured phrases."""
    if not SR_AVAILABLE:
        on_error("SpeechRecognition is not installed.")
        return False
    if _STATE["listening"]:
        return False

    stop_event = threading.Event()
    _STATE.update(listening=True, stop_event=stop_event)

    def worker():
        chunks = []
        try:
            try:
                mic = sr.Microphone()
            except Exception as exc:
                on_error("Microphone unavailable. Check PyAudio and microphone permissions: " + str(exc))
                return

            recognizer = sr.Recognizer()
            on_status("Preparing microphone...")
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                on_status("Listening. Speak; press Space again to finish.")
                while not stop_event.is_set():
                    try:
                        chunks.append(recognizer.listen(source, timeout=1, phrase_time_limit=8))
                    except sr.WaitTimeoutError:
                        continue

            if not chunks:
                on_status("No speech captured. Press Space and try again.")
                return

            on_status("Transcribing speech...")
            phrases = []
            for chunk in chunks:
                try:
                    phrase = recognizer.recognize_google(chunk, language="en-ZA")
                    if phrase:
                        phrases.append(phrase.strip())
                except sr.UnknownValueError:
                    continue

            transcript = " ".join(phrases).strip()
            if transcript:
                on_transcript(transcript)
            else:
                on_status("Speech not understood. Please try again.")
        except sr.RequestError as exc:
            on_error("Google recognition failed; check internet access: " + str(exc))
        except Exception as exc:
            on_error("Dictation error: " + str(exc))
        finally:
            _STATE.update(listening=False, stop_event=None, thread=None)

    thread = threading.Thread(target=worker, daemon=True, name="GUI-Speech-Recognition")
    _STATE["thread"] = thread
    thread.start()
    return True


def stop_dictation():
    if not _STATE["listening"]:
        return False
    event = _STATE.get("stop_event")
    if event:
        event.set()
    return True


def is_listening():
    return bool(_STATE["listening"])
