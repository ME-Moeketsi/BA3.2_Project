import re
import threading
import tkinter as tk

import pyttsx3
import speech_recognition as sr


def start_voice_assistant(root, inputs, on_status=None):
    """
    Voice assistant for the AI triage estimation application.

    - Speaks one question at a time.
    - Waits for the patient's spoken response.
    - Validates answers.
    - Updates the existing Tkinter form.
    - Reports progress through an optional status label.
    """

    stop_event = threading.Event()
    recognizer = sr.Recognizer()

    # --------------------------------------------------
    # STATUS UPDATES
    # --------------------------------------------------

    def status(message):
        print("[VOICE]", message, flush=True)

        if on_status:
            try:
                root.after(
                    0,
                    lambda m=message: on_status(m)
                )
            except tk.TclError:
                pass

    # --------------------------------------------------
    # ANSWER PARSERS
    # --------------------------------------------------

    def parse_sex(answer):
        answer = answer.lower().strip()

        if re.search(r"\b(male|man|boy)\b", answer):
            return "Male"

        if re.search(r"\b(female|woman|girl)\b", answer):
            return "Female"

        return None

    def parse_yes_no(answer):
        answer = answer.lower().strip()

        if re.search(
            r"\b(yes|yeah|yep|true|present)\b",
            answer
        ):
            return "True"

        if re.search(
            r"\b(no|nope|false|absent)\b",
            answer
        ):
            return "False"

        return None

    def parse_number(answer, minimum, maximum):
        answer = answer.lower().strip()

        # First try to recognize a numeric response.
        match = re.search(r"\b\d+(?:\.\d+)?\b", answer)

        if match:
            value = float(match.group())

            if value.is_integer():
                value = int(value)

            if minimum <= value <= maximum:
                return value

        # Support common spoken number words.
        number_words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
            "hundred": 100
        }

        words = re.findall(r"[a-z]+", answer)

        total = 0
        found_number = False

        for word in words:
            if word in number_words:
                found_number = True
                value = number_words[word]

                if value == 100:
                    total = 100
                else:
                    total += value

            elif word == "and" and found_number:
                continue

        if found_number and minimum <= total <= maximum:
            return total

        return None

    def parse_age(answer):
        value = parse_number(answer, 1, 100)

        if value is not None and float(value).is_integer():
            return int(value)

        return None

    def parse_temperature(answer):
        # Compatible with the existing integer Spinbox.
        return parse_number(answer, 1, 100)

    def parse_level(answer):
        answer = answer.lower().strip()

        match = re.search(r"\blevel\s*([1-4])\b", answer)

        if match:
            return "Level " + match.group(1)

        match = re.search(r"\b([1-4])\b", answer)

        if match:
            return "Level " + match.group(1)

        level_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4
        }

        for word, number in level_words.items():
            if re.search(r"\b" + word + r"\b", answer):
                return "Level " + str(number)

        return None

    def parse_complaint(answer):
        answer = answer.strip()

        if len(answer) >= 2:
            return answer

        return None

    # --------------------------------------------------
    # UPDATE EXISTING TKINTER INPUTS
    # --------------------------------------------------

    def update_input(key, value):
        variable = inputs[key]

        if key == "complain":
            # Multiline Text widget
            variable.delete("1.0", "end")
            variable.insert("1.0", value)

        else:
             # StringVar, IntVar, or other Tkinter variable
            variable.set(value)
    def set_input(key, value):
        """
        Update Tkinter widgets on the main GUI thread.
        """

        completed = threading.Event()
        errors = []

        def perform_update():
            try:
                update_input(key, value)
            except Exception as error:
                errors.append(error)
            finally:
                completed.set()

        try:
            root.after(0, perform_update)
        except tk.TclError:
            return False

        completed.wait()

        if errors:
            status(
                f"Could not update {key}: {errors[0]}"
            )
            return False

        return True

    # --------------------------------------------------
    # VOICE INTERVIEW
    # --------------------------------------------------

    def run_interview():
        speaker = None
        com_initialized = False

        try:
            # Initialize Windows COM in the worker thread.
            try:
                import pythoncom

                pythoncom.CoInitialize()
                com_initialized = True

            except ImportError:
                pass

            status("Initializing speech engine...")

            # Keep pyttsx3 version 2.91 installed.
            try:
                speaker = pyttsx3.init("sapi5")
            except Exception:
                speaker = pyttsx3.init()

            speaker.setProperty("rate", 165)
            speaker.setProperty("volume", 1.0)

            # ------------------------------------------
            # SPEAKING FUNCTION
            # ------------------------------------------

            def speak(message):
                if stop_event.is_set():
                    return False

                print(
                    "[VOICE SPEAKING]",
                    message,
                    flush=True
                )

                status("Speaking: " + message)

                try:
                    speaker.say(message)
                    speaker.runAndWait()
                    return not stop_event.is_set()

                except Exception as error:
                    status(
                        "Speech playback error: "
                        + str(error)
                    )
                    print(
                        "[SPEECH ERROR]",
                        repr(error),
                        flush=True
                    )
                    return False

            # ------------------------------------------
            # LISTENING FUNCTION
            # ------------------------------------------

            def listen_for_answer():
                if stop_event.is_set():
                    return None

                try:
                    status("Opening microphone...")

                    with sr.Microphone() as source:
                        status(
                            "Adjusting microphone. "
                            "Please wait..."
                        )

                        recognizer.adjust_for_ambient_noise(
                            source,
                            duration=0.5
                        )

                        if stop_event.is_set():
                            return None

                        status(
                            "Listening now. "
                            "Please speak your answer."
                        )

                        audio = recognizer.listen(
                            source,
                            timeout=10,
                            phrase_time_limit=20
                        )

                    if stop_event.is_set():
                        return None

                    status("Recognizing your answer...")

                    answer = recognizer.recognize_google(
                        audio,
                        language="en-US"
                    ).strip()

                    print(
                        "[VOICE RECOGNIZED]",
                        answer,
                        flush=True
                    )

                    return answer

                except sr.WaitTimeoutError:
                    status(
                        "No speech detected. "
                        "Repeating the question."
                    )
                    return ""

                except sr.UnknownValueError:
                    status(
                        "Speech not understood. "
                        "Please try again."
                    )
                    return ""

                except sr.RequestError as error:
                    status(
                        "Recognition service error: "
                        + str(error)
                    )

                    speak(
                        "Speech recognition is unavailable. "
                        "Please check your internet connection."
                    )

                    return None

                except Exception as error:
                    status(
                        "Microphone error: " + str(error)
                    )

                    speak(
                        "I could not access the microphone. "
                        "Please check your microphone settings."
                    )

                    return None

            # ------------------------------------------
            # ASK A QUESTION AND VALIDATE THE ANSWER
            # ------------------------------------------

            def ask(question, parser):
                while not stop_event.is_set():

                    # Speak the question before opening the mic.
                    if not speak(question):
                        return None

                    if stop_event.is_set():
                        return None

                    answer = listen_for_answer()

                    if answer is None:
                        return None

                    if not answer:
                        continue

                    result = parser(answer)

                    if result is not None:
                        status("Answer accepted.")
                        return result

                    if not speak(
                        "Sorry, I did not understand. "
                        "Please answer again."
                    ):
                        return None

                return None

            # ------------------------------------------
            # INTERVIEW QUESTIONS
            # ------------------------------------------

            questions = [
                (
                    "sex",
                    "What is the patient's sex? "
                    "Please say male or female.",
                    parse_sex
                ),
                (
                    "age",
                    "What is the patient's age in years?",
                    parse_age
                ),
                (
                    "injury",
                    "Is there a visible injury? "
                    "Please say yes or no.",
                    parse_yes_no
                ),
                (
                    "complain",
                    "What is the patient's chief complaint? "
                    "Please describe the main problem.",
                    parse_complaint
                ),
                (
                    "pain",
                    "Is the patient experiencing pain? "
                    "Please say yes or no.",
                    parse_yes_no
                ),
                (
                    "na",
                    "What is the nurse assessment level? "
                    "Please say level one, level two, "
                    "level three, or level four.",
                    parse_level
                ),
                (
                    "bt",
                    "What is the patient's body temperature? "
                    "Please say the numeric temperature.",
                    parse_temperature
                ),
                (
                    "ms",
                    "What is the patient's mental state level? "
                    "Please say level one, level two, "
                    "level three, or level four.",
                    parse_level
                )
            ]

            # ------------------------------------------
            # START THE INTERVIEW
            # ------------------------------------------

            status("Voice interview started.")

            if not speak(
                "Welcome to the patient assessment. "
                "I will ask one question at a time. "
                "Please answer after each question."
            ):
                return

            for key, question, parser in questions:
                if stop_event.is_set():
                    break

                status("Preparing question: " + key)

                result = ask(question, parser)

                if result is None:
                    if not stop_event.is_set():
                        status(
                            "Interview stopped before an answer "
                            "could be recorded. Complete the "
                            "remaining fields manually."
                        )
                    return

                if not set_input(key, result):
                    return

                if not speak("Answer recorded."):
                    return

            # ------------------------------------------
            # INTERVIEW COMPLETE
            # ------------------------------------------

            if not stop_event.is_set():
                status(
                    "Interview complete. "
                    "Please review all answers."
                )

                speak(
                    "The interview is complete. "
                    "Please review all answers on the screen."
                )

        except Exception as error:
            status(
                "Voice assistant error: " + str(error)
            )

            print(
                "[VOICE ERROR]",
                repr(error),
                flush=True
            )

        finally:
            if speaker is not None:
                try:
                    speaker.stop()
                except Exception:
                    pass

            if com_initialized:
                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

            status("Voice assistant session ended.")

    # --------------------------------------------------
    # START THE BACKGROUND THREAD
    # --------------------------------------------------

    threading.Thread(
        target=run_interview,
        daemon=True,
        name="VoiceInterview"
    ).start()

    return stop_event