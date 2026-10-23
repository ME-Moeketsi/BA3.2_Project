import tkinter as tk
from tkinter import ttk, messagebox
import sys
import threading
from pathlib import Path

# The GUI must work whether launched from project root or GUI/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from prediction_service import assess_patient
from NLP.waiting_time_nlp import format_waiting_time

# ==================================================
# 1. MAIN WINDOW
# ==================================================

root = tk.Tk()
root.title("Predict-Care AI | NLP Triage and Waiting Times")
root.geometry("1100x750")
root.minsize(900, 550)
root.configure(bg="#F1F5F9")
root.resizable(True, True)

# ==================================================
# 2. COLORS AND SETTINGS
# ==================================================

BG = "#F1F5F9"
WHITE = "#FFFFFF"
PRIMARY = "#6366F1"
PRIMARY_DARK = "#4F46E5"
TEXT = "#1E293B"
MUTED = "#64748B"
BORDER = "#DCE4F0"
SELECTED_BG = "#EEF2FF"

# Minimum preferred width of each container.
MIN_CARD_WIDTH = 280

# Store input variables and container frames.
inputs = {}
container_cards = []


# ==================================================
# RIGHT-SIDE CONTAINER WITH 3 EQUAL SECTIONS
# ==================================================

right_container = tk.Frame(
    root,
    bg=BG,
    highlightbackground=BORDER,
    highlightthickness=1
)

right_container.place(
    relx=1.0,
    rely=0.5,
    relwidth=0.25,
    relheight=0.90,
    x=-15,
    anchor="e"
)

# Configure three equally sized containers.
right_container.grid_columnconfigure(0, weight=1)

for row in range(3):
    right_container.grid_rowconfigure(row, weight=1, uniform="right_sections")

# Create the three sections.
right_sections = []

for i in range(3):
    section = tk.Frame(
        right_container,
        bg=WHITE,
        highlightbackground=BORDER,
        highlightthickness=1,
        padx=10,
        pady=10
    )

    section.grid(
        row=i,
        column=0,
        sticky="nsew",
        padx=6,
        pady=6
    )

    right_sections.append(section)

# Optional headings for each section.
section_titles = [
    "Patient Summary",
    "Experimental KTAS / Urgency",
    "Estimated Waiting Times"
]

for section, title in zip(right_sections, section_titles):
    tk.Label(
        section,
        text=title,
        font=("Segoe UI", 12, "bold"),
        bg=WHITE,
        fg=TEXT,
        anchor="w"
    ).pack(
        fill="x"
    )


# ==================================================
# PATIENT SUMMARY DISPLAY
# ==================================================

summary_label = tk.Label(
    right_sections[0],
    text="Patient Summary\n\nEnter a chief complaint and click Calculate.",
    font=("Segoe UI", 10),
    bg=WHITE,
    fg=TEXT,
    anchor="nw",
    justify="left",
    wraplength=220
)

summary_label.pack(
    fill="both",
    expand=True,
    anchor="nw",
    padx=5,
    pady=10
)
def resize_summary(event):
    summary_label.configure(
        wraplength=max(100, event.width - 30)
    )

right_sections[0].bind(
    "<Configure>",
    resize_summary
)


# ==================================================
# CALCULATE BUTTON - SECOND RIGHT-SIDE CONTAINER
# ==================================================
calculate_label = tk.Label(
    right_sections[1],
    text="Type a chief complaint and click Calculate.\n\n"
         "The classifier is experimental, not a clinical assessment.",
    font=("Segoe UI", 10), bg=WHITE, fg=TEXT,
    anchor="nw", justify="left", wraplength=220
)

calculate_label.pack(
    fill="both",
    expand=True,
    anchor="nw",
    padx=5,
    pady=10
 )
def resize_calculate(event):
    calculate_label.configure(
        wraplength=max(100, event.width - 30)
    )

right_sections[1].bind(
    "<Configure>",
    resize_calculate
)



calculate_button = tk.Button(
    right_sections[1],
    text="Calculate predictions",
    font=("Segoe UI", 11, "bold"),
    bg=PRIMARY,
    fg=WHITE,
    activebackground=PRIMARY_DARK,
    activeforeground=WHITE,
    relief="flat",
    borderwidth=0,
    cursor="hand2",
    padx=20,
    pady=10,
    command=lambda: calculate_predictions()
)

calculate_button.pack(
    fill="x",
    padx=12,
    pady=12
)






# ==================================================
# VOICE ASSESSMENT BUTTON
# ==================================================

voice_status = tk.Label(
    right_sections[0],
    text="Ready for voice assessment",
    font=("Segoe UI", 9),
    bg=WHITE,
    fg=MUTED,
    wraplength=200
)

voice_status.pack(
    fill="x",
    padx=12,
    pady=8
)


voice_session = None


def update_voice_status(message):
    voice_status.configure(text=message)


def begin_voice_assessment():
    global voice_session

    # Optional feature: speech packages are not required to run predictions.
    try:
        from voice_assistant import start_voice_assistant
    except ImportError as exc:
        messagebox.showinfo(
            "Voice input unavailable",
            f"Optional speech dependencies are not installed: {exc}\n"
            "You can still type patient information and calculate predictions."
        )
        return
    if voice_session is not None and not voice_session.is_set():
        voice_session.set()  # Stop an earlier interview before starting again.
    voice_session = start_voice_assistant(
        root, inputs, on_status=update_voice_status
    )

voice_button = tk.Button(
    right_sections[0],
    text="Start Voice Assessment",
    command=begin_voice_assessment,
    font=("Segoe UI", 10, "bold"),
    bg=PRIMARY,
    fg=WHITE,
    activebackground=PRIMARY_DARK,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    padx=10,
    pady=10
)

voice_button.pack(
    fill="x",
    padx=12,
    pady=8
)

# ==================================================
# 3. SCROLLABLE CONTENT AREA
# ==================================================

content_area = tk.Frame(root, bg=BG)

content_area.place(
    relx=0.0,
    rely=0.0,
    relwidth=0.73,
    relheight=0.90,
    x=12,
    y=12
)

# Canvas makes it possible to scroll through the cards.
canvas = tk.Canvas(
    content_area,
    bg=BG,
    highlightthickness=0,
    borderwidth=0
)

# Vertical scrollbar connected to the canvas.
scrollbar = ttk.Scrollbar(
    content_area,
    orient="vertical",
    command=canvas.yview
)

canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# All containers will be placed inside this frame.
scroll_frame = tk.Frame(canvas, bg=BG)

canvas_window = canvas.create_window(
    (0, 0),
    window=scroll_frame,
    anchor="nw"
)

# ==================================================
# 4. SCROLLING FUNCTIONS
# ==================================================

def update_scroll_region(event=None):
    """Update the area that can be scrolled."""

    canvas.configure(
        scrollregion=canvas.bbox("all")
    )


def resize_content(event):
    """Resize the content and rearrange the containers."""

    canvas.itemconfigure(
        canvas_window,
        width=event.width
    )

    arrange_containers(event.width)


# Update scrolling when the content changes.
scroll_frame.bind(
    "<Configure>",
    update_scroll_region
)

# Rearrange the cards when the window is resized.
canvas.bind(
    "<Configure>",
    resize_content
)


def on_mousewheel(event):
    """Scroll the form with the mouse wheel."""

    if event.num == 4:
        # Linux: scroll upward.
        canvas.yview_scroll(-1, "units")

    elif event.num == 5:
        # Linux: scroll downward.
        canvas.yview_scroll(1, "units")

    elif event.delta:
        # Windows and macOS mouse-wheel support.
        canvas.yview_scroll(
            int(-event.delta / 120),
            "units"
        )


def bind_mousewheel(widget):
    """Enable mouse-wheel scrolling over child widgets."""

    widget.bind("<MouseWheel>", on_mousewheel)
    widget.bind("<Button-4>", on_mousewheel)
    widget.bind("<Button-5>", on_mousewheel)

    for child in widget.winfo_children():
        bind_mousewheel(child)


# ==================================================
# 5. CUSTOM RADIO INDICATOR
# ==================================================

def draw_radio(radio_canvas, selected):
    """Draw a custom radio circle using a Canvas."""

    radio_canvas.delete("all")

    # Draw the outer circle.
    radio_canvas.create_oval(
        3, 3, 19, 19,
        outline=PRIMARY if selected else "#94A3B8",
        width=2,
        fill=WHITE
    )

    # Draw the inner circle when selected.
    if selected:
        radio_canvas.create_oval(
            8, 8, 14, 14,
            outline=PRIMARY,
            fill=PRIMARY
        )


# ==================================================
# 6. REUSABLE CONTAINER CREATOR
# ==================================================

def create_container(container_id, title, subtitle):
    """Create a modern card with a heading and input area."""

    # Main white container.
    card = tk.Frame(
        scroll_frame,
        bg=WHITE,
        highlightbackground=BORDER,
        highlightthickness=1,
        padx=16,
        pady=16
    )

    # Save the card for responsive layout.
    container_cards.append(card)

    card.grid_columnconfigure(0, weight=1)

    # Container title.
    tk.Label(
        card,
        text=title,
        font=("Segoe UI", 15, "bold"),
        bg=WHITE,
        fg=TEXT,
        anchor="w"
    ).grid(
        row=0,
        column=0,
        sticky="ew",
        pady=(0, 6)
    )

    # Container description.
    tk.Label(
        card,
        text=subtitle,
        font=("Segoe UI", 9),
        bg=WHITE,
        fg=MUTED,
        anchor="w",
        justify="left",
        wraplength=300
    ).grid(
        row=1,
        column=0,
        sticky="ew",
        pady=(0, 14)
    )

    # Area where input widgets will be placed.
    body = tk.Frame(card, bg=WHITE)

    body.grid(
        row=2,
        column=0,
        sticky="nsew"
    )

    card.grid_rowconfigure(2, weight=1)

    return card, body


# ==================================================
# 7. CUSTOM RADIO BUTTON GROUP
# ==================================================

def create_radio_group(parent, options, variable):
    """Create a group of clickable custom radio cards."""

    widgets = []

    def refresh():
        """Update the appearance of every radio option."""

        for title, frame, indicator, description in widgets:

            selected = variable.get() == title

            background = (
                SELECTED_BG if selected else WHITE
            )

            # Highlight the selected card.
            frame.configure(
                bg=background,
                highlightbackground=(
                    PRIMARY if selected else BORDER
                )
            )

            indicator.configure(bg=background)
            description.configure(bg=background)

            # Redraw the radio indicator.
            draw_radio(indicator, selected)

    def select_option(title):
        """Save the selected option and refresh the group."""

        variable.set(title)
        refresh()
            # Refresh the radio cards whenever the variable changes,
    # including when the voice assistant updates it.
    variable.trace_add(
        "write",
        lambda *_: refresh()
    )

    for title, description in options:

        # Frame for one radio option.
        option_card = tk.Frame(
            parent,
            bg=WHITE,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=8,
            pady=7,
            cursor="hand2"
        )

        option_card.pack(
            fill="x",
            pady=3
        )

        option_card.grid_columnconfigure(1, weight=1)

        # Custom radio circle.
        indicator = tk.Canvas(
            option_card,
            width=22,
            height=22,
            bg=WHITE,
            highlightthickness=0,
            cursor="hand2"
        )

        indicator.grid(
            row=0,
            column=0,
            rowspan=2,
            padx=(0, 7)
        )

        # Option title.
        title_label = tk.Label(
            option_card,
            text=title,
            font=("Segoe UI", 10, "bold"),
            bg=WHITE,
            fg=TEXT,
            anchor="w",
            cursor="hand2"
        )

        title_label.grid(
            row=0,
            column=1,
            sticky="ew"
        )

        # Option description.
        description_label = tk.Label(
            option_card,
            text=description,
            font=("Segoe UI", 8),
            bg=WHITE,
            fg=MUTED,
            anchor="w",
            cursor="hand2"
        )

        description_label.grid(
            row=1,
            column=1,
            sticky="ew"
        )

        widgets.append(
            (
                title,
                option_card,
                indicator,
                description_label
            )
        )

        # Clicking anywhere on the card selects the option.
        for widget in (
            option_card,
            indicator,
            title_label,
            description_label
        ):
            widget.bind(
                "<Button-1>",
                lambda event, t=title: select_option(t)
            )

    # Display the initial selection.
    refresh()
    # ==================================================
# 8. CONTAINER 1: SEX
# ==================================================

sex_card, sex_body = create_container(
    "Sex",
    "Sex",
    "Choose the Sex of the patient."
)

# Store the selected category.
sex_value = tk.StringVar(value="")
inputs["sex"] = sex_value

# Create four custom radio options.
create_radio_group(
    sex_body,
    [
        ("Male", ""),
        ("Female", ""),

    ],
    sex_value
)


# ==================================================
# 9. CONTAINER 2: Age
# ==================================================

age_card, age_body = create_container(
    "Age",
    "Age of the Patient",
    "Enter the age of the patient."
)

tk.Label(
    age_body,
    text="Age of Patient",
    font=("Segoe UI", 9, "bold"),
    bg=WHITE,
    fg=TEXT
).pack(
    anchor="w",
    pady=(0, 8)
)

# Store the selected quantity.
age_value = tk.StringVar(value="")

# Spinbox lets the user choose a number from 1 to 100.
age_selector = tk.Entry(
    age_body, textvariable=age_value,
    font=("Segoe UI", 11), relief="solid", bd=1, width=10
)

age_selector.pack(
    anchor="w",
    ipady=5
)

inputs["age"] = age_value


# ==================================================
# 10. CONTAINER 3: Injury Status
# ==================================================

Injury_card, Injury_body = create_container(
    "Injury",
    "Injury status",
    "Enter the Patient's Injury Status."
)

# Store the selected billing cycle.
Injury_value = tk.StringVar(value="")
inputs["injury"] = Injury_value

create_radio_group(
    Injury_body,
    [
        ("True", "There is a visible injury"),
        ("False", "There is no visible injury"),

    ],
    Injury_value
)


# ==================================================
# 11. CONTAINER 4: Chief Complain
# ==================================================

Complain_card, Complain_body = create_container(
    "Complain",
    "Chief Complain",
    "Describe the main symptoms in your own words (required by NLP)."
)

# ------------------------------
# Complain field
# ------------------------------

tk.Label(
    Complain_body,
    text="Your message",
    font=("Segoe UI", 9, "bold"),
    bg=WHITE,
    fg=TEXT
).pack(
    anchor="w",
    pady=(0, 7)
)

# A multiline text box for longer messages.
Complain_text = tk.Text(
    Complain_body,
    height=5,
    font=("Segoe UI", 10),
    wrap="word",
    relief="solid",
    bd=1,
    highlightthickness=1,
    highlightbackground=BORDER,
    highlightcolor=PRIMARY,
    padx=8,
    pady=8
)

Complain_text.pack(
    fill="both",
    expand=True
)

inputs["complain"] = Complain_text


# ==================================================
# 12. CONTAINER 5: Pain
# ==================================================

Pain_card, Pain_body = create_container(
    "Pain",
    "Patient's Pain",
    "Confirm if Patient is in pain."
)

# Store the selected Option.
Pain_value = tk.StringVar(value="")
inputs["pain"] = Pain_value

create_radio_group(
    Pain_body,
    [
        ("True", "The patient is in pain"),
        ("False", "The patient is not in pain"),

    ],
    Pain_value
)


# ==================================================
# 13. CONTAINER 6: Nurse Assessment
# ==================================================

NA_card, NA_body = create_container(
    "Nurse Assessment",
    "Nurse Assessment",
    "Enter the Nurse's Assesment ."
)

tk.Label(
    NA_body,
    text="Assessment",
    font=("Segoe UI", 9, "bold"),
    bg=WHITE,
    fg=TEXT
).pack(
    anchor="w",
    pady=(0, 8)
)

# Store the dropdown selection.
NA_value = tk.StringVar(value="")

# Create a dropdown menu.
NA_dropdown = ttk.Combobox(
    NA_body,
    textvariable=NA_value,
    values=[
        "Level 1",
        "Level 2",
        "Level 3",
        "Level 4"
    ],
    state="readonly",
    font=("Segoe UI", 10)
)

NA_dropdown.pack(
    fill="x",
    ipady=5
)

inputs["na"] = NA_value


# ==================================================
# 14. CONTAINER 7: Body Temprature
# ==================================================

BT_card, BT_body = create_container(
    "Temprature",
    "Body Temperature",
    "Record measured body temperature in °C (30–45)."
)

tk.Label(
    BT_body,
    text="Temprature",
    font=("Segoe UI", 9, "bold"),
    bg=WHITE,
    fg=TEXT
).pack(
    anchor="w",
    pady=(0, 8)
)

# Store the selected quantity.
BT_value = tk.StringVar(value="")

# Spinbox lets the user choose a number from 1 to 100.
BT_selector = tk.Entry(
    BT_body, textvariable=BT_value,
    font=("Segoe UI", 11), relief="solid", bd=1, width=10
)

BT_selector.pack(
    anchor="w",
    ipady=5
)

inputs["bt"] = BT_value


# ==================================================
# 15. CONTAINER 8: Mental State
# ==================================================

MS_card, MS_body = create_container(
    "Mental State",
    "Patient's Mental state",
    "Select the patients Mental State."
)

# Store the selected Option.
MS_value = tk.StringVar(value="")
inputs["ms"] = MS_value

create_radio_group(
    MS_body,
    [
        ("Level 1", " The patient is Alert"),
        ("Level 2", "The patient has a Verbal Response"),
        ("Level 3", "The patient responds to pain"),
        ("Level 4", "The patient has no response"),

    ],
    MS_value
)


# ==================================================
# 16. HOSPITAL FLOW + LANGUAGE REQUEST
# ==================================================
operation_card, operation_body = create_container(
    "HospitalFlow",
    "Hospital operations",
    "These values must match the simulation used to train the waiting-time models."
)


def add_operational_input(title, note, variable):
    tk.Label(operation_body, text=title, bg=WHITE, fg=TEXT,
             font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", pady=(8, 2))
    tk.Entry(operation_body, textvariable=variable, font=("Segoe UI", 10),
             relief="solid", bd=1).pack(fill="x", ipady=4)
    tk.Label(operation_body, text=note, bg=WHITE, fg=MUTED,
             font=("Segoe UI", 8), anchor="w", wraplength=280,
             justify="left").pack(fill="x")


arrival_minutes_var = tk.StringVar(value="")
doctors_var = tk.StringVar(value="")
peak_var = tk.StringVar(value="")
request_var = tk.StringVar(value="predict both waiting times")

add_operational_input(
    "Arrival time (simulation minutes)",
    "From 3 to 1107. Not a time of day or a real hospital ETA.",
    arrival_minutes_var
)
add_operational_input(
    "Doctors available",
    "Choose 1–4 (the training-data range).", doctors_var
)
tk.Label(operation_body, text="Peak arrivals?", bg=WHITE, fg=TEXT,
         font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", pady=(8, 2))
ttk.Combobox(operation_body, textvariable=peak_var,
             values=["Yes", "No"], state="readonly").pack(fill="x")
add_operational_input(
    "Ask AI (waiting-time NLP)",
    "Example: 'doctor wait', 'total hospital time', or 'predict both waiting times'.",
    request_var
)

# The third right-side card was reserved but had no content in the original GUI.
waiting_label = tk.Label(
    right_sections[2],
    text="Waiting-time predictions are not calculated yet.\n\n"
         "Enter hospital operational inputs and click Calculate.",
    font=("Segoe UI", 10), bg=WHITE, fg=TEXT,
    anchor="nw", justify="left", wraplength=220,
)
waiting_label.pack(fill="both", expand=True, padx=5, pady=10)
right_sections[2].bind(
    "<Configure>",
    lambda event: waiting_label.configure(wraplength=max(100, event.width - 30))
)

# ==================================================
# 16. RESPONSIVE CONTAINER LAYOUT
# ==================================================

def arrange_containers(available_width):
    """
    Arrange containers according to the available width.

    Narrow window: one column.
    Medium window: two columns.
    Wide window: three columns.
    """

    if not container_cards:
        return

    # Determine how many cards can fit horizontally.
    columns = max(
        1,
        min(
            3,
            int(
                (available_width + 12)
                / (MIN_CARD_WIDTH + 12)
            )
        )
    )

    # Configure the grid columns.
    for column in range(3):
        scroll_frame.grid_columnconfigure(
            column,
            weight=1 if column < columns else 0,
            uniform="cards" if column < columns else ""
        )

    # Place each container in the appropriate row and column.
    for index, card in enumerate(container_cards):

        row = index // columns
        column = index % columns

        card.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=6,
            pady=6
        )

    # Refresh the scrollable area after rearranging.
    root.after_idle(update_scroll_region)
    # ==================================================
# 17. CONTINUE BUTTON ACTION
# ==================================================

def continue_action():
    """Review only supplied fields; the NLP predictor requires the complaint."""
    complaint = inputs["complain"].get("1.0", "end-1c").strip()
    if not complaint:
        messagebox.showwarning("Missing complaint", "Please describe the patient's chief complaint.")
        return None

    age_text = inputs["age"].get().strip()
    bt_text = inputs["bt"].get().strip()
    if age_text:
        try:
            age = int(age_text)
            if not 1 <= age <= 100:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid age", "Enter an age between 1 and 100, or leave it blank.")
            return None
    if bt_text:
        try:
            bt = float(bt_text)
            if not 30 <= bt <= 45:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid temperature", "Enter a temperature between 30 and 45 °C, or leave it blank.")
            return None

    fields = [
        ("Sex", inputs["sex"].get()),
        ("Age", age_text),
        ("Injury", inputs["injury"].get()),
        ("Chief complaint", complaint),
        ("Pain", inputs["pain"].get()),
        ("Nurse assessment", inputs["na"].get()),
        ("Body temperature (°C)", bt_text),
        ("Mental state", inputs["ms"].get()),
    ]
    summary = "\n".join(f"{title}: {value}" for title, value in fields if str(value).strip())
    summary_label.configure(text="Patient Summary\n\n" + summary)
    return {"complaint": complaint, "summary": summary}


# ==================================================
# 18. CONNECT GUI TO NLP CLASSIFIER AND WAIT-TIME REGRESSORS
# ==================================================
def calculate_predictions():
    patient = continue_action()
    if patient is None:
        return

    # An unfilled operations section must never become fabricated numeric data.
    arrival_text = arrival_minutes_var.get().strip()
    doctors_text = doctors_var.get().strip()
    peak_text = peak_var.get().strip()
    if not (arrival_text and doctors_text and peak_text):
        arrival_time = num_doctors = is_peak = None
    else:
        try:
            arrival_time = int(arrival_text)
            num_doctors = int(doctors_text)
            is_peak = {"Yes": 1, "No": 0}[peak_text]
        except (ValueError, KeyError):
            messagebox.showwarning(
                "Invalid hospital input",
                "Enter whole-number arrival minutes, whole-number doctors and Yes/No peak status."
            )
            return

    request_text = request_var.get()
    calculate_button.configure(state="disabled", text="Calculating...")
    calculate_label.configure(
        text="Running saved NLP triage model and hospital waiting-time models...",
        bg=WHITE, fg=TEXT
    )
    waiting_label.configure(text="Calculating available predictions...")

    def worker():
        try:
            output = assess_patient(
                patient["complaint"],
                arrival_time=arrival_time,
                num_doctors=num_doctors,
                is_peak=is_peak,
                waiting_request=request_text,
            )
            error = None
        except Exception as exc:
            output, error = None, exc

        def display():
            calculate_button.configure(state="normal", text="Calculate")
            if error is not None:
                calculate_label.configure(
                    text="Prediction unavailable.\n\n"
                         f"{type(error).__name__}: {error}\n\n"
                         "Missing model files? See GUI/README.md for training steps."
                )
                waiting_label.configure(text="No waiting-time estimate produced.")
                return

            calculate_label.configure(
                text=f"NLP-estimated KTAS: {output.ktas} / 5\n"
                     f"Urgency indication: {output.urgency}\n\n"
                     f"{output.risk_notice}"
            )
            if output.emergency_warning:
                waiting_label.configure(text=output.waiting_message)
                messagebox.showwarning("Urgent safety notice", output.emergency_warning)
                return

            estimates = []
            if output.doctor_minutes is not None and output.waiting_intent in ("doctor", "both"):
                estimates.append(
                    f"Time to doctor: {format_waiting_time(output.doctor_minutes)} "
                    f"({output.doctor_minutes:.1f} min)"
                )
            if output.total_minutes is not None and output.waiting_intent in ("total", "both"):
                estimates.append(
                    f"Total hospital time: {format_waiting_time(output.total_minutes)} "
                    f"({output.total_minutes:.1f} min)"
                )
            if estimates:
                estimates.extend(("", "Simulation estimates only, NOT live queue times. "
                    "Do not delay medical care based on these outputs."))
                waiting_label.configure(text="\n".join(estimates))
            else:
                waiting_label.configure(text=output.waiting_message)

        try:
            root.after(0, display)
        except RuntimeError:
            pass  # Window closed while calculation was in progress.

    threading.Thread(target=worker, daemon=True, name="PredictCareInference").start()


# ==================================================
# 18. FIXED BOTTOM BUTTON
# ==================================================

# This frame is outside the scrollable canvas.
# Therefore, the button stays visible while scrolling.

bottom_area = tk.Frame(
    root,
    bg=BG
)

bottom_area.pack(
    side="bottom",
    fill="x",
    padx=12,
    pady=(5, 16)
)

# Create the Continue button.
continue_button = tk.Button(
    bottom_area,
    text="Preview patient summary",
    command=continue_action,
    font=("Segoe UI", 11, "bold"),
    bg=PRIMARY,
    fg=WHITE,
    activebackground=PRIMARY_DARK,
    activeforeground=WHITE,
    relief="flat",
    borderwidth=0,
    cursor="hand2",
    padx=35,
    pady=10
)

# Keep the button centered horizontally.
continue_button.pack(
    anchor="center"
)


# ==================================================
# 19. INITIALIZE THE APPLICATION
# ==================================================

# Calculate the initial window dimensions.
root.update_idletasks()

# Arrange the containers based on the available width.
arrange_containers(
    canvas.winfo_width()
)

# Enable mouse-wheel scrolling over the content.
bind_mousewheel(scroll_frame)

# Start the Tkinter event loop.
root.mainloop()