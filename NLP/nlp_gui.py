"""
nlp_gui.py

A Unified Flet GUI for Predict-Care AI.
Includes:
  1. KTAS Prediction
  2. Waiting-Time Prediction
  3. Local AI Assistant Chat
  4. Integrated Test

Run with:
    python -m NLP.nlp_gui
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import re
import sys
import threading
import warnings
from datetime import datetime
from pathlib import Path

import flet as ft
import speech_recognition as sr

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Here I resolve the project root from this file, rather than relying on
# whichever directory PowerShell happens to be using when the GUI starts.
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

# Import the backend modules as members of the NLP package. This prevents
# Python from looking for pipeline.py in the project root by mistake.
try:
    from NLP.pipeline import route
    from NLP.waiting_time_nlp import (
        understand_waiting_time_request,
        get_clarification_message,
    )
    from NLP.local_language_engine_expanded import LocalLanguageEngine
    from NLP import chatbot
    from GUI import gui_logic
except ImportError as import_error:
    raise ImportError(
        "Predict-Care AI could not load its backend modules. Check that "
        "NLP/pipeline.py, NLP/waiting_time_nlp.py, "
        "NLP/local_language_engine_expanded.py, NLP/chatbot.py and "
        "GUI/gui_logic.py exist in the project. Original error: "
        f"{import_error}"
    ) from import_error


# ==================================================
# COLORS & STYLES
# ==================================================
BG = "#F1F5F9"
WHITE = "#FFFFFF"
PRIMARY = "#6366F1"
PRIMARY_DARK = "#4F46E5"
TEXT = "#1E293B"
MUTED = "#64748B"
BORDER = "#DCE4F0"
SUCCESS = "#10B981"
ERROR = "#EF4444"

# ==================================================
# SHARED HELPERS
# ==================================================
def get_arrival_time_from_system_clock():
    now = datetime.now()
    return (now.hour * 60 + now.minute) % 240

def _find_focused_control(page) -> object:
    result = [None]
    def walk(ctrl):
        try:
            if getattr(ctrl, "focused", False):
                result[0] = ctrl
                return
            for child in getattr(ctrl, "controls", []) or []:
                walk(child)
            content = getattr(ctrl, "content", None)
            if content is not None: walk(content)
        except Exception: 
            pass
    walk(page)
    return result[0]

class AudioRecorder:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        try:
            self.mic = sr.Microphone(device_index=1)
        except Exception:
            self.mic = sr.Microphone()
        self.is_recording = False
        self.stop_event = threading.Event()
        self.audio_data = None

    def start(self):
        self.is_recording = True
        self.stop_event.clear()
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            try:
                self.audio_data = self.recognizer.listen(
                    source, timeout=None, phrase_time_limit=None, snowboy_configuration=None
                )
            except Exception:
                pass

    def stop(self):
        self.is_recording = False
        self.stop_event.set()

    def transcribe(self):
        if not self.audio_data:
            return None
        try:
            return self.recognizer.recognize_google(self.audio_data, language="en-ZA")
        except sr.UnknownValueError:
            return "ERROR: Speech could not be understood."
        except sr.RequestError as e:
            return f"ERROR: Speech service error: {e}"
        except Exception as e:
            return f"ERROR: {e}"


# ==================================================
# MAIN APPLICATION
# ==================================================
def main(page: ft.Page):
    page.title = "Predict-Care AI"
    page.bgcolor = BG
    page.window.width = 1200
    page.window.height = 900
    page.window.min_width = 900
    page.window.min_height = 700
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    engine = None
    try:
        engine = LocalLanguageEngine()
    except Exception as e:
        print("Failed to load Language Engine:", e)

    active_hooks = {"space": None}

    # ---------------- VIEW: HOME MENU ----------------
    def show_home(e=None):
        active_hooks["space"] = None
        page.controls.clear()
        
        def menu_card(title, desc, icon, on_click):
            return ft.Container(
                content=ft.Column([
                    ft.Icon(icon, size=40, color=PRIMARY),
                    ft.Text(title, size=20, weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Text(desc, size=12, color=MUTED, text_align=ft.TextAlign.CENTER),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                bgcolor=WHITE,
                padding=30,
                border_radius=12,
                border=ft.Border.all(1, BORDER),
                width=250,
                height=220,
                ink=True,
                on_click=on_click,
                shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.with_opacity(0.05, "black"))
            )

        menu_row = ft.Row(
            controls=[
                menu_card("KTAS Triage", "Classify a chief complaint into an emergency acuity level.", ft.Icons.MEDICAL_SERVICES, lambda e: show_ktas_view()),
                menu_card("Waiting Time", "Predict doctor wait and total hospital time.", ft.Icons.ACCESS_TIME, lambda e: show_wait_view()),
                menu_card("AI Assistant", "Chat with the local NLP language engine.", ft.Icons.CHAT, lambda e: show_chat_view()),
                menu_card("Integrated Test", "Full pipeline test (Traditional + NLP + DL).", ft.Icons.TROUBLESHOOT, lambda e: show_test_view()),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
            wrap=True
        )

        page.add(
            ft.Column([
                ft.Container(height=40),
                ft.Text("Predict-Care AI", size=36, weight=ft.FontWeight.BOLD, color=PRIMARY),
                ft.Text("Select a module to begin", size=16, color=MUTED),
                ft.Container(height=40),
                menu_row
            ], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        )
        page.update()

    # ==================================================
    # VIEW: KTAS PREDICTION
    # ==================================================
    def show_ktas_view():
        page.controls.clear()
        
        complaint_field = ft.TextField(label="Chief Complaint", multiline=True, min_lines=3, max_lines=5, text_size=14)
        result_text = ft.Text("", size=14, color=TEXT)
        result_box = ft.Container(content=result_text, padding=20, bgcolor=WHITE, border=ft.Border.all(1, BORDER), border_radius=10, visible=False)
        
        mic_btn = ft.IconButton(icon=ft.Icons.MIC, icon_color=WHITE, bgcolor=ERROR, tooltip="Speak complaint (Space)", width=50, height=50)
        recorder = AudioRecorder()
        recording_thread = None

        def _set_loading(msg="Processing..."):
            result_box.visible = True
            result_text.value = msg
            result_text.color = MUTED
            page.update()

        def _set_result(msg, is_error=False):
            result_box.visible = True
            result_text.value = msg
            result_text.color = ERROR if is_error else TEXT
            page.update()

        def on_submit(e=None):
            complaint = complaint_field.value.strip()
            if not complaint:
                _set_result("Please enter a chief complaint.", True)
                return
            
            _set_loading("Running KTAS model...")
            
            def worker():
                try:
                    res = route(complaint)
                    if res["type"] != "ktas":
                        _set_result(f"Prediction failed:\n{res.get('message', 'Unknown error')}", True)
                        return
                    
                    r = res["result"]
                    out = (
                        f"Chief Complaint: {r['chief_complaint']}\n"
                        f"Predicted KTAS: {r['predicted_ktas']}\n"
                        f"Emergency Category: {r['emergency_category']}\n\n"
                        "Note: Decision-support only. Not a clinical assessment."
                    )
                    _set_result(out)
                except Exception as ex:
                    _set_result(f"ERROR: {ex}", True)

            threading.Thread(target=worker, daemon=True).start()

        def toggle_mic(e=None):
            nonlocal recording_thread
            if not recorder.is_recording:
                recorder.is_recording = True
                mic_btn.icon = ft.Icons.STOP
                mic_btn.bgcolor = PRIMARY
                _set_loading("Listening... Click 🛑 or press Space to stop.")
                def record_worker(): recorder.start()
                recording_thread = threading.Thread(target=record_worker, daemon=True)
                recording_thread.start()
            else:
                recorder.stop()
                mic_btn.icon = ft.Icons.MIC
                mic_btn.bgcolor = ERROR
                _set_loading("Transcribing audio...")
                def transcribe_worker():
                    text = recorder.transcribe()
                    if text and not text.startswith("ERROR:"):
                        complaint_field.value = text
                        _set_result(f"Heard: \"{text}\"")
                        on_submit(None)
                    else:
                        _set_result(text or "No audio heard.", True)
                    page.update()
                threading.Thread(target=transcribe_worker, daemon=True).start()

        mic_btn.on_click = toggle_mic
        active_hooks["space"] = toggle_mic

        page.add(
            ft.Column([
                ft.Row([
                    ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=show_home),
                    ft.Text("KTAS Prediction", size=24, weight=ft.FontWeight.BOLD)
                ]),
                ft.Container(height=20),
                ft.Container(
                    content=ft.Column([
                        ft.Text("Enter Chief Complaint", size=16, weight=ft.FontWeight.BOLD),
                        complaint_field,
                        ft.Row([
                            ft.Button("Predict KTAS", bgcolor=PRIMARY, color=WHITE, on_click=on_submit, height=50, expand=True),
                            mic_btn
                        ], spacing=10),
                    ], spacing=15),
                    padding=20, bgcolor=WHITE, border=ft.Border.all(1, BORDER), border_radius=10
                ),
                ft.Container(height=20),
                result_box
            ], expand=True, scroll=ft.ScrollMode.AUTO)
        )
        page.update()

    # ==================================================
    # VIEW: WAITING TIME
    # ==================================================
    def show_wait_view():
        page.controls.clear()

        priority = ft.Dropdown(label="Priority", options=[ft.dropdown.Option("1 - Low"), ft.dropdown.Option("2 - Medium"), ft.dropdown.Option("3 - High")], value="1 - Low")
        doctors = ft.TextField(label="Number of Doctors", value="2", keyboard_type=ft.KeyboardType.NUMBER)
        peak = ft.Dropdown(label="Peak Arrival?", options=[ft.dropdown.Option("1", "Yes"), ft.dropdown.Option("0", "No")], value="0")
        
        arrival = get_arrival_time_from_system_clock()
        arrival_lbl = ft.Text(f"Arrival time (auto): {arrival} mins into 240-min window", size=12, color=MUTED)

        request = ft.Dropdown(label="What to predict?", options=[ft.dropdown.Option("doctor", "Doctor Wait"), ft.dropdown.Option("total", "Total Time"), ft.dropdown.Option("both", "Both")], value="both")

        result_text = ft.Text("", size=14, color=TEXT)
        result_box = ft.Container(content=result_text, padding=20, bgcolor=WHITE, border=ft.Border.all(1, BORDER), border_radius=10, visible=False)
        
        voice_status = ft.Text("", size=12, color=MUTED)
        mic_btn = ft.IconButton(icon=ft.Icons.MIC, icon_color=WHITE, bgcolor=ERROR, tooltip="Speak request (Space)", width=50, height=50)
        
        recorder = AudioRecorder()
        recording_thread = None

        def _set_loading(msg="Processing..."):
            result_box.visible = True
            result_text.value = msg
            result_text.color = MUTED
            page.update()

        def _set_result(msg, is_error=False):
            result_box.visible = True
            result_text.value = msg
            result_text.color = ERROR if is_error else TEXT
            page.update()

        def on_submit(e=None):
            try:
                p_val = int(priority.value.split(" ")[0])
                d_val = int(doctors.value)
                i_peak = int(peak.value)
            except ValueError:
                _set_result("Invalid numeric input.", True)
                return

            patient_info = {"Priority": p_val, "ArrivalTime": arrival, "NumDoctors": d_val, "IsPeakArrival": i_peak}
            _set_loading("Running prediction...")

            def worker():
                try:
                    res = route(request.value, patient_info=patient_info)
                    if res["type"] == "error":
                        _set_result(f"Error: {res['message']}", True)
                        return
                    
                    out = f"Intent: {res.get('intent', 'unknown')}\n"
                    out += f"{res.get('message', '')}\n\n"
                    if "doctor_minutes" in res: out += f"Doctor Wait: {res['doctor_minutes']:.1f} mins\n"
                    if "total_minutes" in res: out += f"Total Time: {res['total_minutes']:.1f} mins"
                    _set_result(out)
                except Exception as ex:
                    _set_result(f"ERROR: {ex}", True)

            threading.Thread(target=worker, daemon=True).start()

        def toggle_mic(e=None):
            nonlocal recording_thread
            if not recorder.is_recording:
                recorder.is_recording = True
                mic_btn.icon = ft.Icons.STOP
                mic_btn.bgcolor = PRIMARY
                voice_status.value = "Listening... Click 🛑 or press Space to stop."
                voice_status.color = ERROR
                page.update()
                def record_worker(): recorder.start()
                recording_thread = threading.Thread(target=record_worker, daemon=True)
                recording_thread.start()
            else:
                recorder.stop()
                mic_btn.icon = ft.Icons.MIC
                mic_btn.bgcolor = ERROR
                voice_status.value = "Transcribing..."
                voice_status.color = MUTED
                page.update()
                def transcribe_worker():
                    text = recorder.transcribe()
                    if text and not text.startswith("ERROR:"):
                        voice_status.value = f"Heard: \"{text}\""
                        voice_status.color = SUCCESS
                        intent = understand_waiting_time_request(text)
                        if intent in ("doctor", "total", "both"): request.value = intent
                        elif intent == "clarify": request.value = "both"
                        on_submit(None)
                    else:
                        voice_status.value = text or "No audio heard."
                        voice_status.color = ERROR
                    page.update()
                threading.Thread(target=transcribe_worker, daemon=True).start()

        mic_btn.on_click = toggle_mic
        active_hooks["space"] = toggle_mic

        page.add(
            ft.Column([
                ft.Row([
                    ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=show_home),
                    ft.Text("Waiting-Time Prediction", size=24, weight=ft.FontWeight.BOLD)
                ]),
                ft.Container(height=20),
                ft.Container(
                    content=ft.Column([
                        ft.Text("Patient Information", size=16, weight=ft.FontWeight.BOLD),
                        ft.Row([priority, peak], spacing=10),
                        ft.Row([doctors]),
                        arrival_lbl,
                        ft.Divider(),
                        ft.Text("Request", size=16, weight=ft.FontWeight.BOLD),
                        ft.Row([request, mic_btn], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        voice_status,
                        ft.Button("Predict Wait Time", bgcolor=PRIMARY, color=WHITE, on_click=on_submit, height=50)
                    ], spacing=15),
                    padding=20, bgcolor=WHITE, border=ft.Border.all(1, BORDER), border_radius=10
                ),
                ft.Container(height=20),
                result_box
            ], expand=True, scroll=ft.ScrollMode.AUTO)
        )
        page.update()

    # ==================================================
    # VIEW: AI CHAT ASSISTANT
    # ==================================================
    def show_chat_view():
        """Conversational assistant with session context and live wait predictions."""
        active_hooks["space"] = None
        page.controls.clear()

        chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        input_field = ft.TextField(
            hint_text="Type or use the microphone...", expand=True,
            on_submit=lambda e: send_msg()
        )
        # Voice controls belong only to the AI Assistant view.
        voice_state = {"listening": False, "speech_enabled": False}
        chat_recognizer = sr.Recognizer()
        mic_status = ft.Text("", size=12, color=MUTED)
        mic_btn = ft.IconButton(
            icon=ft.Icons.MIC, tooltip="Speak a message",
            icon_color=WHITE, bgcolor=PRIMARY, width=48, height=48
        )
        speaker_btn = ft.IconButton(
            icon=ft.Icons.VOLUME_OFF, tooltip="Enable spoken responses",
            icon_color=PRIMARY, width=48, height=48
        )

        # Here I keep recent turns and the most recent waiting-time inputs for
        # follow-up messages such as "yes", "what about one doctor?", or "and off peak?".
        conversation = []
        wait_context = {
            "Priority": 2,
            "ArrivalTime": get_arrival_time_from_system_clock(),
            "NumDoctors": 2,
            "IsPeakArrival": 0,
        }
        last_assistant_message = {"text": ""}
        busy = {"value": False}

        def add_msg(text, is_user=False):
            bg = PRIMARY if is_user else WHITE
            color = WHITE if is_user else TEXT
            align = ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
            chat_list.controls.append(
                ft.Row([
                    ft.Container(
                        content=ft.Text(str(text), color=color, size=14, selectable=True),
                        padding=12,
                        bgcolor=bg,
                        border_radius=ft.BorderRadius(8, 8, 0 if is_user else 8, 8 if is_user else 0),
                        border=ft.Border.all(1, BORDER) if not is_user else None,
                    )
                ], alignment=align)
            )
            page.update()

        def is_waiting_question(text):
            normalized = text.lower()
            return any(term in normalized for term in (
                "wait", "waiting time", "how long", "peak", "off peak", "off-peak",
                "doctor", "priority level", "hospital time", "queue", "arrive at",
                "what about one", "what if there are", "what if i come"
            ))

        def update_wait_context(text):
            """Update only the values explicitly mentioned; keep previous values otherwise."""
            msg = text.lower()
            found_any = False

            # Priority: accept "priority 2", "priority level 2", or "level 2" in a wait context.
            m = re.search(r"\bpriority(?:\s+level)?\s*(?:of\s*)?(1|2|3)\b", msg)
            if m:
                wait_context["Priority"] = int(m.group(1)); found_any = True
            elif is_waiting_question(msg):
                m = re.search(r"\blevel\s*(1|2|3)\b", msg)
                if m:
                    wait_context["Priority"] = int(m.group(1)); found_any = True

            m = re.search(r"\b(\d{1,3})\s*(?:doctors?|physicians?)\b", msg)
            if m:
                wait_context["NumDoctors"] = max(1, min(100, int(m.group(1)))); found_any = True
            else:
                m = re.search(r"\b(?:with|using|have)\s+(?:only\s+)?(\d{1,2})\s+(?:available\s+)?(?:doctor|physician)", msg)
                if m:
                    wait_context["NumDoctors"] = max(1, min(100, int(m.group(1)))); found_any = True
                elif re.search(r"\b(one|a single) doctor\b", msg):
                    wait_context["NumDoctors"] = 1; found_any = True
                elif re.search(r"\btwo doctors\b", msg):
                    wait_context["NumDoctors"] = 2; found_any = True

            m = re.search(r"\b(?:at|time(?:\s+is)?|arrive(?:\s+at)?)\s*(\d{1,3})\b", msg)
            if m and ("peak" in msg or "arriv" in msg or "wait" in msg):
                wait_context["ArrivalTime"] = max(0, min(239, int(m.group(1)))); found_any = True
            else:
                m = re.search(r"\b(\d{1,3})\s*(?:minutes?|mins?)\s*(?:into|after)\b", msg)
                if m:
                    wait_context["ArrivalTime"] = max(0, min(239, int(m.group(1)))); found_any = True

            if any(term in msg for term in ("off peak", "off-peak", "not peak", "outside peak", "non peak", "non-peak")):
                wait_context["IsPeakArrival"] = 0; found_any = True
            elif "peak" in msg:
                # Avoid interpreting "not peak" as peak (handled above).
                if not any(term in msg for term in ("off peak", "off-peak", "not peak", "non-peak", "non peak")):
                    wait_context["IsPeakArrival"] = 1; found_any = True

            return found_any

        def perform_wait_prediction(text):
            """Call the trained regression models through the shared NLP pipeline."""
            update_wait_context(text)
            request_text = text
            intent = understand_waiting_time_request(request_text)
            if intent not in ("doctor", "total", "both"):
                # Natural follow-ups can be terse; preserve the previous prediction type.
                intent = "both"
            info = dict(wait_context)
            result = route(intent, patient_info=info)
            if result.get("type") == "error":
                return (
                    "I understood the waiting-time question, but the prediction pipeline returned an error. "
                    "No estimate has been invented.\n\n" + result.get("message", "Unknown prediction error")
                )
            if result.get("needs_clarification"):
                return result.get("message", "Would you like doctor waiting time, total hospital time, or both?")

            parts = ["Waiting-time estimate from the trained model:"]
            if "doctor_minutes" in result:
                parts.append(f"• Doctor wait: {result['doctor_minutes']:.1f} minutes")
            if "total_minutes" in result:
                parts.append(f"• Total hospital time: {result['total_minutes']:.1f} minutes")
            parts.append(
                "\nInputs used: priority level {Priority}, arrival minute {ArrivalTime}/240, "
                "{NumDoctors} doctor(s), {peak}.".format(
                    **info, peak="peak arrivals" if info["IsPeakArrival"] else "off-peak arrivals"
                )
            )
            parts.append("This is a model estimate for decision support, not a guarantee of actual waiting time.")
            return "\n".join(parts)

        def answer_message(text):
            msg = text.strip()
            normalized = re.sub(r"[^a-z0-9 ]", "", msg.lower()).strip()

            if normalized in ("/reset", "reset conversation", "start over", "clear context"):
                conversation.clear()
                wait_context.update({
                    "Priority": 2,
                    "ArrivalTime": get_arrival_time_from_system_clock(),
                    "NumDoctors": 2,
                    "IsPeakArrival": 0,
                })
                return "Conversation context has been reset. What would you like to know?"

            # If a user answers "yes" to an immediately preceding assistant question,
            # carry that confirmation forward instead of classifying it as an unrelated intent.
            if normalized in ("yes", "yeah", "yep", "correct", "thats right", "yes please"):
                previous = last_assistant_message["text"].lower()
                if "doctor waiting time, total hospital time, or both" in previous:
                    return perform_wait_prediction("both waiting time")
                if "waiting-time estimate from the trained model" in previous or "inputs used:" in previous:
                    return perform_wait_prediction("both waiting time")
                if "would you like" in previous or previous.rstrip().endswith("?"):
                    return "Thanks — I’ll use that confirmation for the question we were discussing. Please add the detail you want me to apply, such as the number of doctors or whether it is peak time."

            if normalized in ("no", "nope", "not really") and last_assistant_message["text"].endswith("?"):
                return "Okay. Tell me what you would prefer, and I’ll update the request."

            if normalized in ("why", "why is that", "explain that", "what do you mean", "can you explain", "tell me more"):
                previous = last_assistant_message["text"].strip()
                if previous:
                    return (
                        "You’re asking about my previous response. It was: \"" + previous + "\"\n\n"
                        "I can explain the model inputs and limitations, or refine the estimate if you change a value."
                    )

            if is_waiting_question(msg) or (conversation and conversation[-1]["kind"] == "waiting" and any(
                term in normalized for term in ("what about", "instead", "and if", "how about", "one doctor", "two doctors", "peak", "off peak", "same but")
            )):
                return perform_wait_prediction(msg)

            if engine is None:
                return "The local language engine did not load. Please check the startup error in the terminal."
            response, should_exit = engine.respond(msg)
            if should_exit:
                return response
            return response

        def speak_response(text):
            """Speak assistant replies only when the speaker toggle is enabled."""
            if not voice_state["speech_enabled"] or not text:
                return
            try:
                import pyttsx3
                speaker = pyttsx3.init()
                speaker.setProperty("rate", 175)
                speaker.say(text)
                speaker.runAndWait()
                speaker.stop()
            except Exception as ex:
                mic_status.value = f"Speech output unavailable: {ex}. Install pyttsx3."
                mic_status.color = ERROR
                page.update()

        def toggle_speaker(e=None):
            voice_state["speech_enabled"] = not voice_state["speech_enabled"]
            speaker_btn.icon = (
                ft.Icons.VOLUME_UP if voice_state["speech_enabled"]
                else ft.Icons.VOLUME_OFF
            )
            speaker_btn.tooltip = (
                "Disable spoken responses" if voice_state["speech_enabled"]
                else "Enable spoken responses"
            )
            mic_status.value = (
                "Spoken responses enabled." if voice_state["speech_enabled"]
                else "Spoken responses disabled."
            )
            mic_status.color = SUCCESS if voice_state["speech_enabled"] else MUTED
            page.update()

        def toggle_chat_mic(e=None):
            if voice_state["listening"]:
                return
            voice_state["listening"] = True
            mic_btn.disabled = True
            mic_btn.icon = ft.Icons.MIC
            mic_status.value = "Listening — speak after the microphone opens..."
            mic_status.color = MUTED
            page.update()

            def listen_worker():
                try:
                    # Prefer the default microphone. This avoids relying on a
                    # hard-coded device index that differs between computers.
                    with sr.Microphone() as source:
                        chat_recognizer.adjust_for_ambient_noise(source, duration=0.35)
                        audio = chat_recognizer.listen(
                            source, timeout=6, phrase_time_limit=15
                        )
                    heard = chat_recognizer.recognize_google(audio, language="en-ZA")
                    input_field.value = heard
                    mic_status.value = f"Recognised: {heard}"
                    mic_status.color = SUCCESS
                    page.update()
                    send_msg()
                except sr.WaitTimeoutError:
                    mic_status.value = "No speech detected. Click the microphone and try again."
                    mic_status.color = ERROR
                    page.update()
                except sr.UnknownValueError:
                    mic_status.value = "I could not understand that audio. Please try again."
                    mic_status.color = ERROR
                    page.update()
                except sr.RequestError as ex:
                    mic_status.value = f"Speech recognition service error: {ex}"
                    mic_status.color = ERROR
                    page.update()
                except Exception as ex:
                    mic_status.value = f"Microphone error: {ex}"
                    mic_status.color = ERROR
                    page.update()
                finally:
                    voice_state["listening"] = False
                    mic_btn.disabled = False
                    mic_btn.icon = ft.Icons.MIC
                    page.update()

            threading.Thread(target=listen_worker, daemon=True).start()

        mic_btn.on_click = toggle_chat_mic
        speaker_btn.on_click = toggle_speaker

        def send_msg(e=None):
            if busy["value"]:
                return
            text = (input_field.value or "").strip()
            if not text:
                return
            input_field.value = ""
            add_msg(text, is_user=True)
            conversation.append({"role": "user", "text": text, "kind": "waiting" if is_waiting_question(text) else "general"})
            conversation[:] = conversation[-12:]
            busy["value"] = True

            def worker():
                try:
                    response = answer_message(text)
                    last_assistant_message["text"] = response
                    conversation.append({"role": "assistant", "text": response, "kind": "assistant"})
                    conversation[:] = conversation[-12:]
                    add_msg(response, is_user=False)
                    speak_response(response)
                except Exception as ex:
                    response = f"I couldn't complete that request. No estimate was invented.\nError: {type(ex).__name__}: {ex}"
                    last_assistant_message["text"] = response
                    add_msg(response, is_user=False)
                    speak_response(response)
                finally:
                    busy["value"] = False

            threading.Thread(target=worker, daemon=True).start()

        add_msg(
            "Hello! I am the Predict-Care AI local assistant. I can predict doctor waiting time and total hospital time using the trained model. "
            "Try: ‘How long will I wait at peak, arrival minute 120, priority level 2, with 2 doctors?’ "
            "I keep the current conversation context for follow-up questions. Type /help for supported commands or /reset to start a new topic."
        )

        page.add(
            ft.Column([
                ft.Row([
                    ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=show_home),
                    ft.Text("AI Assistant", size=24, weight=ft.FontWeight.BOLD, color=TEXT)
                ]),
                ft.Container(height=10),
                ft.Container(content=chat_list, expand=True, bgcolor=BG, padding=10, border=ft.Border.all(1, BORDER), border_radius=10),
                ft.Row([
                    input_field,
                    mic_btn,
                    speaker_btn,
                    ft.IconButton(
                        icon=ft.Icons.SEND, tooltip="Send message",
                        bgcolor=PRIMARY, icon_color=WHITE, on_click=send_msg
                    )
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                mic_status
            ], expand=True)
        )
        page.update()

    # ==================================================
    # VIEW: INTEGRATED TEST (Chatbot Logic)
    # ==================================================
    def show_test_view():
        active_hooks["space"] = None
        page.controls.clear()

        # Load KTAS options dynamically
        try:
            options = chatbot.load_ktas_options()
        except Exception as ex:
            page.add(ft.Text(f"Error loading options: {ex}", color=ERROR))
            return

        # --- Setup form fields with Human-Readable Labels ---
        group = ft.Dropdown(label="Group (1 or 2)", options=[ft.dropdown.Option("1"), ft.dropdown.Option("2")], value="1")
        sex = ft.Dropdown(label="Sex (1 = male, 2 = female)", options=[ft.dropdown.Option("1"), ft.dropdown.Option("2")], value="1")
        age = ft.TextField(label="Age", value="30", keyboard_type=ft.KeyboardType.NUMBER)
        pat_hr = ft.TextField(label="Patients per Hour", value="10", keyboard_type=ft.KeyboardType.NUMBER)

        # Use gui_logic to generate the exact same display strings
        arr_disp = gui_logic.display_options_for("Arrival mode", options["Arrival mode"])
        inj_disp = gui_logic.display_options_for("Injury", options["Injury"])
        men_disp = gui_logic.display_options_for("Mental", options["Mental"])
        pain_disp = gui_logic.display_options_for("Pain", options["Pain"])

        arr_mode = ft.Dropdown(label="Mode of Arrival", options=[ft.dropdown.Option(v) for v in arr_disp])
        injury = ft.Dropdown(label="Injury", options=[ft.dropdown.Option(v) for v in inj_disp])
        mental = ft.Dropdown(label="Mental Status", options=[ft.dropdown.Option(v) for v in men_disp])
        pain = ft.Dropdown(label="Pain", options=[ft.dropdown.Option(v) for v in pain_disp])
        
        nrs = ft.TextField(label="NRS Pain Score (0-10)", value="0", keyboard_type=ft.KeyboardType.NUMBER)
        sbp = ft.TextField(label="Systolic BP (SBP)", value="120", keyboard_type=ft.KeyboardType.NUMBER)
        dbp = ft.TextField(label="Diastolic BP (DBP)", value="80", keyboard_type=ft.KeyboardType.NUMBER)
        hr = ft.TextField(label="Heart Rate (HR)", value="75", keyboard_type=ft.KeyboardType.NUMBER)
        rr = ft.TextField(label="Respiratory Rate (RR)", value="16", keyboard_type=ft.KeyboardType.NUMBER)
        bt = ft.TextField(label="Body Temperature (C)", value="37", keyboard_type=ft.KeyboardType.NUMBER)
        sat = ft.TextField(label="Oxygen Saturation (%)", value="98", keyboard_type=ft.KeyboardType.NUMBER)
        
        complaint = ft.TextField(label="Chief Complaint", value="Headache", multiline=True)

        num_docs = ft.TextField(label="Number of Available Doctors", value="2", keyboard_type=ft.KeyboardType.NUMBER)
        peak_limit = ft.TextField(label="Peak Duration (mins)", value="240", keyboard_type=ft.KeyboardType.NUMBER)
        horizon = ft.TextField(label="Forecast Horizon (hrs)", value="3", keyboard_type=ft.KeyboardType.NUMBER)

        result_text = ft.Text("", size=14, color=TEXT, font_family="Consolas")
        result_box = ft.Container(content=result_text, padding=20, bgcolor=WHITE, border=ft.Border.all(1, BORDER), border_radius=10, visible=False)

        def _set_loading(msg="Running..."):
            result_box.visible = True
            result_text.value = msg
            result_text.color = MUTED
            page.update()

        def _set_result(msg, is_error=False):
            result_box.visible = True
            result_text.value = msg
            result_text.color = ERROR if is_error else TEXT
            page.update()

        def on_submit(e=None):
            try:
                # When submitting, extract the first part before " =" to get the raw value
                patient_data = {
                    "Group": int(group.value),
                    "Sex": int(sex.value),
                    "Age": int(age.value),
                    "Patients number per hour": int(pat_hr.value),
                    "Arrival mode": arr_mode.value.split(" = ")[0],
                    "Injury": injury.value.split(" = ")[0],
                    "Mental": mental.value.split(" = ")[0],
                    "Pain": pain.value.split(" = ")[0],
                    "NRS_pain": float(nrs.value),
                    "SBP": float(sbp.value),
                    "DBP": float(dbp.value),
                    "HR": float(hr.value),
                    "RR": float(rr.value),
                    "BT": float(bt.value),
                    "Saturation": float(sat.value),
                    "Chief_complain": complaint.value,
                }
                n_docs = int(num_docs.value)
                p_lim = int(peak_limit.value)
                h_lim = int(horizon.value)
            except Exception as ex:
                _set_result(f"Invalid input: {ex}", True)
                return

            _set_loading("Running full pipeline (Trad + NLP + DL)...")

            def worker():
                try:
                    output = chatbot.run_integrated_test(patient_data, n_docs, p_lim, h_lim)
                    _set_result(output)
                except Exception as ex:
                    _set_result(f"ERROR: {ex}", True)

            threading.Thread(target=worker, daemon=True).start()

        # --- Build Layout ---
        form_col1 = ft.Column([group, sex, age, pat_hr, arr_mode, injury])
        form_col2 = ft.Column([mental, pain, nrs, sbp, dbp, hr])
        form_col3 = ft.Column([rr, bt, sat, num_docs, peak_limit, horizon])

        page.add(
            ft.Column([
                ft.Row([
                    ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=show_home),
                    ft.Text("Integrated Test (Chatbot Backend)", size=24, weight=ft.FontWeight.BOLD)
                ]),
                ft.Container(height=20),
                ft.Container(
                    content=ft.Column([
                        ft.Text("Patient Clinical Information", size=16, weight=ft.FontWeight.BOLD),
                        ft.Row([form_col1, form_col2, form_col3], spacing=15, alignment=ft.MainAxisAlignment.START),
                        complaint,
                        ft.Divider(),
                        ft.Button("Run Test", bgcolor=PRIMARY, color=WHITE, on_click=on_submit, height=50, width=200),
                    ], spacing=15),
                    padding=20, bgcolor=WHITE, border=ft.Border.all(1, BORDER), border_radius=10
                ),
                ft.Container(height=20),
                result_box
            ], expand=True, scroll=ft.ScrollMode.AUTO)
        )
        page.update()

    # ---------------- Global Keyboard Hook ----------------
    def on_keyboard(e: ft.KeyboardEvent):
        if e.key == " ":
            focused = _find_focused_control(page)
            typing = isinstance(focused, ft.TextField) and not focused.read_only
            if not typing and active_hooks["space"] is not None:
                active_hooks["space"](None)
                return "break"

    page.on_keyboard_event = on_keyboard

    # Start at Home
    show_home()


if __name__ == "__main__":
    ft.run(main)