"""
nlp_gui.py

A Unified Flet GUI for Predict-Care AI.
Includes:
  1. KTAS Prediction
  2. Waiting-Time Prediction
  3. Local AI Assistant Chat
  4. Integrated Test

Run with:
    flet run nlp/nlp_gui.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import threading
import time
import flet as ft
from datetime import datetime
import speech_recognition as sr

# Import your existing backend logic
try:
    from pipeline import route
    from waiting_time_nlp import understand_waiting_time_request, get_clarification_message
    from local_language_engine_expanded import LocalLanguageEngine
    
    # Import the newly moved chatbot logic
    from NLP import chatbot
    # Import the GUI logic to reuse its display formatting
    from GUI import gui_logic
except ImportError as e:
    print("Warning: Could not import backend modules. Ensure you run from project root.", e)

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
        active_hooks["space"] = None
        page.controls.clear()

        chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        input_field = ft.TextField(hint_text="Type a message...", expand=True, on_submit=lambda e: send_msg())

        def add_msg(text, is_user=False):
            bg = PRIMARY if is_user else WHITE
            color = WHITE if is_user else TEXT
            align = ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
            
            chat_list.controls.append(
                ft.Row([
                    ft.Container(
                        content=ft.Text(text, color=color, size=14),
                        padding=12,
                        bgcolor=bg,
                        border_radius=ft.BorderRadius(8, 8, 0 if is_user else 8, 8 if is_user else 0),
                        border=ft.Border.all(1, BORDER) if not is_user else None,
                    )
                ], alignment=align)
            )
            page.update()

        def send_msg():
            text = input_field.value.strip()
            if not text: return
            input_field.value = ""
            add_msg(text, is_user=True)
            if engine is None:
                add_msg("System Error: Language engine not loaded.")
                return
            def worker():
                try:
                    response, _ = engine.respond(text)
                    add_msg(response, is_user=False)
                except Exception as ex:
                    add_msg(f"Error: {ex}", is_user=False)
            threading.Thread(target=worker, daemon=True).start()

        add_msg("Hello! I am the Predict-Care AI local assistant. Type /help to see what I can do.")

        page.add(
            ft.Column([
                ft.Row([
                    ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=show_home),
                    ft.Text("AI Assistant", size=24, weight=ft.FontWeight.BOLD)
                ]),
                ft.Container(height=10),
                ft.Container(content=chat_list, expand=True, bgcolor=BG, padding=10, border=ft.Border.all(1, BORDER), border_radius=10),
                ft.Row([input_field, ft.IconButton(icon=ft.Icons.SEND, bgcolor=PRIMARY, icon_color=WHITE, on_click=lambda e: send_msg())])
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