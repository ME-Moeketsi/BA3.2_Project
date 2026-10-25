"""
GUI.py

Tkinter user interface for the AI Triage Estimation tool.

This file contains NO model code. All prediction work is
delegated to gui_logic.py so the UI and the model layer
stay separated.

Run via GUI_Pipeline.run_gui(), or directly with:
    python GUI.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Optional voice module
try:
    from voice_assistant import start_voice_assistant
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False
    start_voice_assistant = None

# Model / prediction layer
import gui_logic


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
VOICE_GREEN = "#10B981"
VOICE_GREEN_DARK = "#059669"

MIN_CARD_WIDTH = 300


# ==================================================
# MAIN ENTRY
# ==================================================

def create_window():
    """
    Build and return the main Tk window with the full GUI
    already laid out. GUI_Pipeline.run_gui() calls this
    and then runs root.mainloop().
    """
    root = tk.Tk()
    root.title("AI Triage Estimation")
    root.geometry("1200x780")
    root.minsize(700, 500)
    root.configure(bg=BG)
    root.resizable(True, True)

    # Shared state
    inputs = {}
    container_cards = []

    # ---- Right panel first so its widgets are created ----
    right = _build_right_panel(root, inputs)
    output_text = right["output_text"]
    voice_status = right["voice_status"]
    calculate_button = right["calculate_button"]
    voice_button = right["voice_button"]

    # ---- Left scrollable form ----
    form = _build_form_panel(root, inputs, container_cards)
    canvas = form["canvas"]
    scroll_frame = form["scroll_frame"]

    # ---- Wire calculate button ----
    def calculate_action():
        _on_calculate(
            inputs,
            output_text,
            calculate_button,
        )

    calculate_button.configure(command=calculate_action)

    # ---- Wire voice button ----
    def voice_action():
        _on_voice(
            root,
            inputs,
            voice_button,
            voice_status,
        )

    voice_button.configure(command=voice_action)

    # ---- Responsive layout ----
    def arrange_containers(available_width):
        if not container_cards:
            return
        columns = max(
            1,
            min(3, int((available_width + 12) / (MIN_CARD_WIDTH + 12))),
        )
        for col in range(3):
            scroll_frame.grid_columnconfigure(
                col,
                weight=1 if col < columns else 0,
                uniform="cards" if col < columns else "",
            )
        for i, card in enumerate(container_cards):
            card.grid(
                row=i // columns,
                column=i % columns,
                sticky="nsew", padx=6, pady=6,
            )
        root.after_idle(
            lambda: canvas.configure(scrollregion=canvas.bbox("all"))
        )

    def resize_content(event):
        canvas.itemconfigure(
            form["canvas_window"], width=event.width
        )
        arrange_containers(event.width)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.bind("<Configure>", resize_content)

    # ---- Mouse wheel ----
    def on_mousewheel(event):
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        elif event.delta:
            canvas.yview_scroll(int(-event.delta / 120), "units")

    def bind_wheel(widget):
        widget.bind("<MouseWheel>", on_mousewheel)
        widget.bind("<Button-4>", on_mousewheel)
        widget.bind("<Button-5>", on_mousewheel)
        for child in widget.winfo_children():
            bind_wheel(child)

    root.update_idletasks()
    arrange_containers(canvas.winfo_width())
    bind_wheel(scroll_frame)

    return root


# ==================================================
# RIGHT PANEL
# ==================================================

def _build_right_panel(root, inputs):
    container = tk.Frame(
        root, bg=WHITE,
        highlightbackground=BORDER, highlightthickness=1,
    )
    container.place(
        relx=1.0, rely=0.5,
        relwidth=0.30, relheight=0.90,
        x=-15, anchor="e",
    )

    tk.Label(
        container, text="Prediction Output",
        font=("Segoe UI", 14, "bold"),
        bg=WHITE, fg=TEXT,
    ).pack(side="top", anchor="w", padx=12, pady=(12, 6))

    # Button bar packed BEFORE output_text so it reserves space
    button_bar = tk.Frame(container, bg=WHITE)
    button_bar.pack(side="bottom", fill="x", padx=10, pady=(4, 10))

    voice_status = tk.Label(
        container, text="",
        font=("Segoe UI", 9), bg=WHITE, fg=MUTED,
        wraplength=280, justify="left",
    )
    voice_status.pack(side="bottom", fill="x", padx=12, pady=(0, 4))

    output_frame = tk.Frame(container, bg=WHITE)
    output_frame.pack(side="top", fill="both", expand=True, padx=6, pady=4)

    scroll = tk.Scrollbar(output_frame, orient="vertical")
    scroll.pack(side="right", fill="y")

    output_text = tk.Text(
        output_frame,
        wrap="word",
        font=("Consolas", 10),
        bg=WHITE, fg=TEXT,
        relief="flat", bd=0,
        padx=8, pady=8,
        yscrollcommand=scroll.set,
        state="disabled",
    )
    output_text.pack(side="left", fill="both", expand=True)
    scroll.config(command=output_text.yview)

    _set_output(
        output_text,
        "Fill in the patient form on the left,\n"
        "then click Calculate.\n\n"
        "The output will show:\n"
        "  - Traditional ML KTAS\n"
        "  - NLP KTAS\n"
        "  - Deep Learning KTAS\n"
        "  - Combined KTAS and probabilities\n"
        "  - Operational Priority\n"
        "  - Peak / Non-peak status\n"
        "  - Predicted doctor waiting time",
    )

    calculate_button = tk.Button(
        button_bar,
        text="Calculate",
        font=("Segoe UI", 10, "bold"),
        bg=PRIMARY, fg=WHITE,
        activebackground=PRIMARY_DARK, activeforeground=WHITE,
        relief="flat", cursor="hand2",
        padx=10, pady=9,
    )
    calculate_button.pack(side="left", fill="x", expand=True, padx=(0, 4))

    voice_button = tk.Button(
        button_bar,
        text="Voice" if VOICE_AVAILABLE else "Voice (unavailable)",
        font=("Segoe UI", 10, "bold"),
        bg=VOICE_GREEN, fg=WHITE,
        activebackground=VOICE_GREEN_DARK, activeforeground=WHITE,
        relief="flat", cursor="hand2",
        padx=10, pady=9,
        state="normal" if VOICE_AVAILABLE else "disabled",
    )
    voice_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

    return {
        "container": container,
        "output_text": output_text,
        "voice_status": voice_status,
        "calculate_button": calculate_button,
        "voice_button": voice_button,
    }


# ==================================================
# FORM PANEL
# ==================================================

def _build_form_panel(root, inputs, container_cards):
    content_area = tk.Frame(root, bg=BG)
    content_area.place(
        relx=0.0, rely=0.0,
        relwidth=0.68, relheight=0.90,
        x=12, y=12,
    )

    canvas = tk.Canvas(
        content_area, bg=BG,
        highlightthickness=0, borderwidth=0,
    )
    scrollbar = ttk.Scrollbar(
        content_area, orient="vertical", command=canvas.yview
    )
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    scroll_frame = tk.Frame(canvas, bg=BG)
    canvas_window = canvas.create_window(
        (0, 0), window=scroll_frame, anchor="nw"
    )

    # Load options via the logic layer
    options, err = gui_logic.safe_load_ktas_options()
    if err:
        print("WARNING loading KTAS options:", err)

    # ---------- Demographics ----------
    _, body = _create_card(
        scroll_frame, container_cards,
        "Demographics",
        "Basic patient information used by all models.",
    )

    inputs["Group"] = tk.StringVar(value="1")
    _combobox(body, "Group (1 or 2)", inputs["Group"], ["1", "2"])

    inputs["Sex"] = tk.StringVar(value="1")
    _combobox(body, "Sex (1 = male, 2 = female)",
              inputs["Sex"], ["1", "2"])

    inputs["Age"] = tk.StringVar(value="30")
    _spinbox(body, "Age", inputs["Age"], 0, 120)

    # ---------- Vitals ----------
    _, body = _create_card(
        scroll_frame, container_cards,
        "Vitals",
        "Numeric measurements used by the ML and DL models.",
    )

    inputs["SBP"] = tk.StringVar(value="120")
    _spinbox(body, "Systolic BP (SBP)", inputs["SBP"], 0, 300)

    inputs["DBP"] = tk.StringVar(value="80")
    _spinbox(body, "Diastolic BP (DBP)", inputs["DBP"], 0, 200)

    inputs["HR"] = tk.StringVar(value="75")
    _spinbox(body, "Heart Rate (HR)", inputs["HR"], 0, 250)

    inputs["RR"] = tk.StringVar(value="16")
    _spinbox(body, "Respiratory Rate (RR)", inputs["RR"], 0, 80)

    inputs["BT"] = tk.StringVar(value="37")
    _spinbox(body, "Body Temperature (C)", inputs["BT"],
             25, 45, step=0.1)

    inputs["Saturation"] = tk.StringVar(value="98")
    _spinbox(body, "Oxygen Saturation (%)",
             inputs["Saturation"], 0, 100)

    # ---------- Clinical Categories ----------
    _, body = _create_card(
        scroll_frame, container_cards,
        "Clinical Categories",
        "Categorical values loaded directly from the training dataset.",
    )

    # Build display-labelled option lists for each categorical field.
    arrival_display = gui_logic.display_options_for(
        "Arrival mode", options["Arrival mode"]
    )
    injury_display = gui_logic.display_options_for(
        "Injury", options["Injury"]
    )
    mental_display = gui_logic.display_options_for(
        "Mental", options["Mental"]
    )
    pain_display = gui_logic.display_options_for(
        "Pain", options["Pain"]
    )

    inputs["Arrival mode"] = tk.StringVar(value=arrival_display[0])
    _combobox(body, "Mode of Arrival",
              inputs["Arrival mode"], arrival_display)

    inputs["Injury"] = tk.StringVar(value=injury_display[0])
    _combobox(body, "Injury", inputs["Injury"], injury_display)

    inputs["Mental"] = tk.StringVar(value=mental_display[0])
    _combobox(body, "Mental Status",
              inputs["Mental"], mental_display)

    inputs["Pain"] = tk.StringVar(value=pain_display[0])
    _combobox(body, "Pain", inputs["Pain"], pain_display)

    inputs["NRS_pain"] = tk.StringVar(value="0")
    _spinbox(body, "NRS Pain Score (0-10)",
             inputs["NRS_pain"], 0, 10, step=0.1)

    # ---------- Chief Complaint ----------
    _, body = _create_card(
        scroll_frame, container_cards,
        "Chief Complaint",
        "Free-text reason for the visit (used by the NLP model).",
    )

    tk.Label(
        body, text="Chief Complaint",
        font=("Segoe UI", 9, "bold"), bg=WHITE, fg=TEXT,
    ).pack(anchor="w", pady=(0, 6))

    cc_text = tk.Text(
        body, height=5, font=("Segoe UI", 10), wrap="word",
        relief="solid", bd=1, highlightthickness=1,
        highlightbackground=BORDER, highlightcolor=PRIMARY,
        padx=8, pady=8,
    )
    cc_text.pack(fill="both", expand=True)
    inputs["Chief_complain"] = cc_text

    # ---------- Operational ----------
    _, body = _create_card(
        scroll_frame, container_cards,
        "Operational Information",
        "Used by the patient-flow waiting-time model.",
    )

    inputs["NumDoctors"] = tk.StringVar(value="2")
    _combobox(body, "Number of Available Doctors",
              inputs["NumDoctors"], ["1", "2", "3", "4"])

    inputs["PeakLimit"] = tk.StringVar(value="240")
    _combobox(body, "Peak Duration (minutes)",
              inputs["PeakLimit"], ["120", "240", "360"])

    inputs["Patients number per hour"] = tk.StringVar(value="10")
    _spinbox(body, "Patients per Hour",
             inputs["Patients number per hour"], 1, 200)

    return {
        "content_area": content_area,
        "canvas": canvas,
        "scroll_frame": scroll_frame,
        "canvas_window": canvas_window,
    }


# ==================================================
# WIDGET HELPERS
# ==================================================

def _create_card(parent, store, title, subtitle):
    card = tk.Frame(
        parent, bg=WHITE,
        highlightbackground=BORDER, highlightthickness=1,
        padx=16, pady=16,
    )
    store.append(card)

    card.grid_columnconfigure(0, weight=1)

    tk.Label(
        card, text=title,
        font=("Segoe UI", 14, "bold"),
        bg=WHITE, fg=TEXT, anchor="w",
    ).grid(row=0, column=0, sticky="ew", pady=(0, 4))

    tk.Label(
        card, text=subtitle,
        font=("Segoe UI", 9),
        bg=WHITE, fg=MUTED,
        anchor="w", justify="left", wraplength=300,
    ).grid(row=1, column=0, sticky="ew", pady=(0, 10))

    body = tk.Frame(card, bg=WHITE)
    body.grid(row=2, column=0, sticky="nsew")
    card.grid_rowconfigure(2, weight=1)
    return card, body


def _spinbox(parent, label, var, from_, to, step=1):
    tk.Label(
        parent, text=label,
        font=("Segoe UI", 9, "bold"), bg=WHITE, fg=TEXT,
    ).pack(anchor="w", pady=(0, 4))
    tk.Spinbox(
        parent, from_=from_, to=to, increment=step,
        textvariable=var, font=("Segoe UI", 11),
        relief="solid", bd=1, width=12,
    ).pack(anchor="w", ipady=4, pady=(0, 10))


def _combobox(parent, label, var, values):
    tk.Label(
        parent, text=label,
        font=("Segoe UI", 9, "bold"), bg=WHITE, fg=TEXT,
    ).pack(anchor="w", pady=(0, 4))
    ttk.Combobox(
        parent, textvariable=var, values=values,
        state="readonly", font=("Segoe UI", 10),
    ).pack(fill="x", ipady=4, pady=(0, 10))


def _set_output(output_text, message):
    output_text.configure(state="normal")
    output_text.delete("1.0", "end")
    output_text.insert("1.0", message)
    output_text.configure(state="disabled")


# ==================================================
# ACTION HANDLERS
# ==================================================

def _collect_raw_values(inputs):
    return {
        "Group": inputs["Group"].get(),
        "Sex": inputs["Sex"].get(),
        "Age": inputs["Age"].get(),
        "Patients number per hour":
            inputs["Patients number per hour"].get(),
        "Arrival mode": inputs["Arrival mode"].get(),
        "Injury": inputs["Injury"].get(),
        "Mental": inputs["Mental"].get(),
        "Pain": inputs["Pain"].get(),
        "NRS_pain": inputs["NRS_pain"].get(),
        "SBP": inputs["SBP"].get(),
        "DBP": inputs["DBP"].get(),
        "HR": inputs["HR"].get(),
        "RR": inputs["RR"].get(),
        "BT": inputs["BT"].get(),
        "Saturation": inputs["Saturation"].get(),
        "Chief_complain":
            inputs["Chief_complain"].get("1.0", "end-1c").strip(),
    }


def _on_calculate(inputs, output_text, calculate_button):
    try:
        patient_data = gui_logic.build_patient_data(
            _collect_raw_values(inputs)
        )
    except (ValueError, tk.TclError) as e:
        messagebox.showwarning(
            "Invalid Input",
            "Please check your inputs:\n" + str(e),
        )
        return

    if not patient_data["Chief_complain"]:
        messagebox.showwarning(
            "Missing Complaint",
            "Please enter the chief complaint.",
        )
        return

    num_doctors = int(inputs["NumDoctors"].get())
    peak_limit = int(inputs["PeakLimit"].get())

    _set_output(output_text, "Running predictions...\nPlease wait.")
    calculate_button.configure(state="disabled")

    def worker():
        try:
            results = gui_logic.run_full_pipeline(
                patient_data, num_doctors, peak_limit
            )
            text = gui_logic.format_results(results)
            output_text.after(0, lambda: _set_output(output_text, text))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            msg = "ERROR during prediction:\n\n" + str(e) + "\n\n" + tb
            output_text.after(0, lambda: _set_output(output_text, msg))
        finally:
            output_text.after(
                0, lambda: calculate_button.configure(state="normal")
            )

    threading.Thread(target=worker, daemon=True).start()


def _on_voice(root, inputs, voice_button, voice_status):
    if not VOICE_AVAILABLE:
        messagebox.showinfo(
            "Voice Unavailable",
            "The voice_assistant module could not be loaded.",
        )
        return
    voice_button.configure(state="disabled")
    start_voice_assistant(
        root, inputs,
        on_status=lambda msg: voice_status.configure(text=msg),
    )


# ==================================================
# DIRECT-RUN SUPPORT
# ==================================================

if __name__ == "__main__":
    window = create_window()
    window.mainloop()