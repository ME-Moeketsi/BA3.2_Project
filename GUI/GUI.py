"""
gui.py

Flet user interface for the AI Triage Estimation tool.

Run with:
    flet run gui/gui.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
# Force the working directory to the project root (parent of the GUI folder)
# This ensures Models/triage_model.pkl is always found regardless of where flet is run from.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import flet as ft

import gui_logic
import gui_speech_logic as speech_logic

# ==================================================
# COLORS
# ==================================================
BG = "#F1F5F9"
WHITE = "#FFFFFF"
PRIMARY = "#6366F1"
PRIMARY_DARK = "#4F46E5"
TEXT = "#1E293B"
MUTED = "#64748B"
BORDER = "#DCE4F0"
VOICE_ACTIVE = "#EF4444"
VOICE_IDLE = "#10B981"

# ==================================================
# MAIN ENTRY
# ==================================================

def main(page: ft.Page):
    page.title = "AI Triage Estimation"
    page.bgcolor = BG
    page.window.width = 1300
    page.window.height = 900
    page.window.min_width = 1000
    page.window.min_height = 700
    page.padding = 16
    page.theme_mode = ft.ThemeMode.LIGHT

    inputs = {}

    # ---------------- Right panel (Output) ----------------
    output_field = ft.TextField(
        multiline=True,
        read_only=True,
        expand=True,
        text_size=13,
        text_style=ft.TextStyle(font_family="Consolas"),
        border=ft.InputBorder.NONE,
        filled=True,
        fill_color=WHITE,
        value=(
            "Fill in the patient form on the left,\n"
            "then click Submit (or press Enter).\n\n"
            "Press Space to dictate fields by voice.\n"
            "Press Space again to stop.\n\n"
            "The output will show:\n"
            "  - Traditional ML KTAS\n"
            "  - NLP KTAS\n"
            "  - Deep Learning KTAS\n"
            "  - Combined KTAS and probabilities\n"
            "  - Operational Priority\n"
            "  - Peak / Non-peak status\n"
            "  - Predicted doctor waiting time\n"
            "  - Time-series demand forecast\n"
            "  - Consistency check"
        ),
    )

    voice_status = ft.Text(
        "Space = voice dictation, Enter = submit",
        size=11,
        color=MUTED,
    )

    submit_button = ft.Button(
        content="Submit",
        bgcolor=PRIMARY,
        color=WHITE,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=ft.Padding.symmetric(vertical=14, horizontal=10),
            bgcolor={"": PRIMARY, "hovered": PRIMARY_DARK},
        ),
        width=float("inf"),
        on_click=lambda e: page.run_task(
            _on_submit_async, page, inputs, output_field, submit_button
        ),
    )

    right_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("Prediction Output", size=18, weight=ft.FontWeight.BOLD, color=TEXT),
                ft.Container(
                    content=output_field,
                    expand=True,
                    bgcolor=WHITE,
                    border=ft.Border.all(1, BORDER),
                    border_radius=8,
                    padding=8,
                ),
                voice_status,
                submit_button,
            ],
            spacing=12,
            expand=True,
        ),
        width=380,
        padding=16,
        bgcolor=WHITE,
        border=ft.Border.all(1, BORDER),
        border_radius=12,
    )

    # ---------------- Left panel (Form) ----------------
    options, err = gui_logic.safe_load_ktas_options()
    if err:
        print("WARNING loading KTAS options:", err)

    # Initialize controls
    inputs["Group"] = ft.Dropdown(label="Group (1 or 2)", options=[ft.dropdown.Option("1"), ft.dropdown.Option("2")], value="1", dense=True)
    inputs["Sex"] = ft.Dropdown(label="Sex (1=male, 2=female)", options=[ft.dropdown.Option("1"), ft.dropdown.Option("2")], value="1", dense=True)
    inputs["Age"] = _number_field("Age", "30")

    inputs["SBP"] = _number_field("Systolic BP (SBP)", "120")
    inputs["DBP"] = _number_field("Diastolic BP (DBP)", "80")
    inputs["HR"] = _number_field("Heart Rate (HR)", "75")
    inputs["RR"] = _number_field("Respiratory Rate (RR)", "16")
    inputs["BT"] = _number_field("Body Temperature (C)", "37")
    inputs["Saturation"] = _number_field("Oxygen Saturation (%)", "98")

    inputs["Arrival mode"] = _dropdown("Mode of Arrival", gui_logic.display_options_for("Arrival mode", options["Arrival mode"]))
    inputs["Injury"] = _dropdown("Injury", gui_logic.display_options_for("Injury", options["Injury"]))
    inputs["Mental"] = _dropdown("Mental Status", gui_logic.display_options_for("Mental", options["Mental"]))
    inputs["Pain"] = _dropdown("Pain", gui_logic.display_options_for("Pain", options["Pain"]))
    inputs["NRS_pain"] = _number_field("NRS Pain Score (0-10)", "0")

    inputs["NumDoctors"] = ft.Dropdown(label="Number of Doctors", options=[ft.dropdown.Option(v) for v in ["1", "2", "3", "4"]], value="2", dense=True)
    inputs["PeakLimit"] = ft.Dropdown(label="Peak Duration (mins)", options=[ft.dropdown.Option(v) for v in ["120", "240", "360"]], value="240", dense=True)
    inputs["Patients number per hour"] = _number_field("Patients per Hour", "10")
    inputs["TSHorizon"] = _number_field("Forecast Horizon (hrs)", "3")

    inputs["Chief_complain"] = ft.TextField(
        label="Chief Complaint",
        multiline=True,
        min_lines=3,
        max_lines=5,
        text_size=14,
        border_color=BORDER,
        focused_border_color=PRIMARY,
    )

    # Build manual 2-column layout using Rows and Columns
    col1 = ft.Column(
        controls=[
            _card("Demographics", "Basic patient information.", [inputs["Group"], inputs["Sex"], inputs["Age"]]),
            _card("Clinical Categories", "Categorical values.", [inputs["Arrival mode"], inputs["Injury"], inputs["Mental"], inputs["Pain"], inputs["NRS_pain"]]),
        ],
        spacing=16,
        expand=True,
    )
    
    col2 = ft.Column(
        controls=[
            _card("Vitals", "Numeric measurements.", [inputs["SBP"], inputs["DBP"], inputs["HR"], inputs["RR"], inputs["BT"], inputs["Saturation"]]),
            _card("Operational Info", "Patient-flow parameters.", [inputs["NumDoctors"], inputs["PeakLimit"], inputs["Patients number per hour"], inputs["TSHorizon"]]),
        ],
        spacing=16,
        expand=True,
    )

    chief_complaint_card = _card("Chief Complaint", "Free-text reason (NLP model).", [inputs["Chief_complain"]])

    form_container = ft.Column(
        controls=[
            ft.Row(
                controls=[col1, col2],
                spacing=16,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            chief_complaint_card,
        ],
        spacing=16,
    )

    left_panel = ft.Container(
        content=ft.Column([form_container], scroll=ft.ScrollMode.AUTO, expand=True),
        expand=True,
        padding=ft.Padding.only(right=8)
    )

    # ---------------- Assemble Page ----------------
    page.add(
        ft.Row(
            [left_panel, right_panel],
            expand=True,
            spacing=16,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
    )

    # ---------------- Keyboard bindings ----------------
    def on_keyboard(e: ft.KeyboardEvent):
        focused = _find_focused_control(page)
        typing = isinstance(focused, ft.TextField) and not focused.read_only

        if e.key == " " and not typing:
            _toggle_dictation(page, inputs, voice_status)
        elif e.key == "Enter" and (not typing or focused is inputs.get("Chief_complain")):
            page.run_task(_on_submit_async, page, inputs, output_field, submit_button)

    page.on_keyboard_event = on_keyboard

    def on_close(_e):
        try: speech_logic.stop_dictation()
        except Exception: pass

    page.on_close = on_close


# ==================================================
# WIDGET HELPERS
# ==================================================

def _number_field(label: str, value: str) -> ft.TextField:
    return ft.TextField(
        label=label, value=value, keyboard_type=ft.KeyboardType.NUMBER, dense=True,
        border_color=BORDER, focused_border_color=PRIMARY,
    )

def _dropdown(label: str, values: list) -> ft.Dropdown:
    return ft.Dropdown(
        label=label, options=[ft.dropdown.Option(v) for v in values],
        value=values[0] if values else None, dense=True,
        border_color=BORDER, focused_border_color=PRIMARY,
    )

def _card(title: str, subtitle: str, children: list) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(title, size=16, weight=ft.FontWeight.BOLD, color=TEXT),
                ft.Text(subtitle, size=10, color=MUTED),
                ft.Divider(height=1, color=BORDER),
                *children,
            ],
            spacing=8,
        ),
        padding=16,
        bgcolor=WHITE,
        border=ft.Border.all(1, BORDER),
        border_radius=10,
    )

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
        except Exception: pass
    walk(page)
    return result[0]

# ==================================================
# ACTION HANDLERS
# ==================================================

def _collect_raw_values(inputs) -> dict:
    return {
        "Group": inputs["Group"].value,
        "Sex": inputs["Sex"].value,
        "Age": inputs["Age"].value,
        "Patients number per hour": inputs["Patients number per hour"].value,
        "Arrival mode": inputs["Arrival mode"].value,
        "Injury": inputs["Injury"].value,
        "Mental": inputs["Mental"].value,
        "Pain": inputs["Pain"].value,
        "NRS_pain": inputs["NRS_pain"].value,
        "SBP": inputs["SBP"].value,
        "DBP": inputs["DBP"].value,
        "HR": inputs["HR"].value,
        "RR": inputs["RR"].value,
        "BT": inputs["BT"].value,
        "Saturation": inputs["Saturation"].value,
        "Chief_complain": (inputs["Chief_complain"].value or "").strip(),
    }

async def _on_submit_async(page, inputs, output_field, submit_button):
    try:
        patient_data = gui_logic.build_patient_data(_collect_raw_values(inputs))
    except (ValueError, TypeError) as e:
        await _show_dialog(page, "Invalid Input", "Please check your inputs:\n" + str(e))
        return

    if not patient_data["Chief_complain"]:
        await _show_dialog(page, "Missing Complaint", "Please enter the chief complaint.")
        return

    submit_button.disabled = True
    output_field.value = "Running predictions...\nPlease wait."
    page.update()

    try:
        results = await asyncio.to_thread(
            gui_logic.run_full_pipeline,
            patient_data,
            int(inputs["NumDoctors"].value),
            int(inputs["PeakLimit"].value),
            ts_horizon=int(inputs["TSHorizon"].value),
        )
        output_field.value = gui_logic.format_results(results)
    except Exception as e:
        import traceback
        output_field.value = f"ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
    finally:
        submit_button.disabled = False
        page.update()

async def _show_dialog(page, title: str, body: str):
    dlg = ft.AlertDialog(
        title=ft.Text(title),
        content=ft.Text(body),
        actions=[ft.TextButton("OK", on_click=lambda e: page.pop_dialog())],
    )
    page.show_dialog(dlg)

def _toggle_dictation(page, inputs, voice_status):
    if speech_logic.is_listening():
        speech_logic.stop_dictation()
        voice_status.value = "Processing..."
        voice_status.color = MUTED
        page.update()
        return

    if not speech_logic.speech_available():
        voice_status.value = "Voice unavailable."
        voice_status.color = MUTED
        page.update()
        return

    voice_status.value = "Listening..."
    voice_status.color = VOICE_ACTIVE
    page.update()

    def on_transcript(t): page.run_task(_handle_transcript_async, page, inputs, voice_status, t)
    def on_status(m): voice_status.value = m; voice_status.color = MUTED; page.update()
    def on_error(m): voice_status.value = "Error: " + m; voice_status.color = MUTED; page.update()

    if not speech_logic.start_dictation(on_transcript=on_transcript, on_status=on_status, on_error=on_error):
        voice_status.value = "Could not start."
        voice_status.color = MUTED
        page.update()

async def _handle_transcript_async(page, inputs, voice_status, transcript):
    options, _ = gui_logic.safe_load_ktas_options()
    updates = speech_logic.parse_dictation(transcript, options)
    if not updates:
        voice_status.value = f'Heard "{transcript}" but no fields recognised.'
        voice_status.color = MUTED
    else:
        for k, v in updates.items():
            if k in inputs: inputs[k].value = v
        voice_status.value = f"Updated {len(updates)} field(s)."
        voice_status.color = VOICE_IDLE
    page.update()

if __name__ == "__main__":
    ft.run(main)