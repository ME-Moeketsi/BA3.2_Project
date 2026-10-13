import tkinter as tk
from tkinter import ttk, messagebox

# ==================================================
# 1. MAIN WINDOW
# ==================================================

root = tk.Tk()
root.title("AI triage estimation")
root.geometry("1100x750")
root.minsize(350, 450)
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
# 3. SCROLLABLE CONTENT AREA
# ==================================================

content_area = tk.Frame(root, bg=BG)
content_area.pack(
    side="top",
    fill="both",
    expand=True,
    padx=12,
    pady=(12, 5)
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
sex_value = tk.StringVar(value="Personal")
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
age_value = tk.IntVar(value=1)

# Spinbox lets the user choose a number from 1 to 100.
age_selector = tk.Spinbox(
    age_body,
    from_=1,
    to=100,
    textvariable=age_value,
    font=("Segoe UI", 11),
    relief="solid",
    bd=1,
    width=10
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
    "Enter Paitiens Chief complain."
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
    "Nurse Assesment",
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
    "Body Temprature",
    "Enter the patient's body Temprature."
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
BT_value = tk.IntVar(value=1)

# Spinbox lets the user choose a number from 1 to 100.
BT_selector = tk.Spinbox(
    BT_body,
    from_=1,
    to=100,
    textvariable=BT_value,
    font=("Segoe UI", 11),
    relief="solid",
    bd=1,
    width=10
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
        ("Level 3", "The patient has aPain response"),
        ("Level 4", "The patient has no response"),

    ],
    MS_value
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
    """Collect, validate, and display the form information."""

    # ----------------------------------------------
    # Read the radio button selections.
    # ----------------------------------------------

    sex = inputs["sex"].get()
    injury = inputs["injury"].get()
    pain = inputs["pain"].get()
    ms = inputs["ms"].get()

    # ----------------------------------------------
    # Read the Chief Complain.
    # ----------------------------------------------

    complain = inputs["complain"].get("1.0", "end-1c").strip()
   

 
    # ----------------------------------------------
    # Read the dropdown selection.
    # ----------------------------------------------

    na = inputs["na"].get()

    # ----------------------------------------------
    # Validate the numbers.
    # ----------------------------------------------

    try:
        age = int(inputs["age"].get())

        if not 1 <= age <= 100:
            raise ValueError

    except (ValueError, tk.TclError):
        messagebox.showwarning(
            "Invalid Quantity",
            "Please enter a quantity between 1 and 100."
        )
        return
    try:
             bt = int(inputs["bt"].get())
     
             if not 1 <= bt <= 100:
                 raise ValueError
     
    except (ValueError, tk.TclError):
             messagebox.showwarning(
                 "Invalid Quantity",
                 "Please enter a quantity between 1 and 100."
             )
             return
    
  
    # ----------------------------------------------
    # Check required radio button selections.
    # ----------------------------------------------

    missing = []

    if not sex:
        missing.append("sex")

    if not injury:
        missing.append("injury")

    if not pain:
        missing.append("pain")
        
    if not ms:
        missing.append("ms")

    if missing:
        messagebox.showwarning(
            "Incomplete Form",
            "Please complete the following selections:\n\n"
            + "\n".join(missing)
        )
        return

 

    # ----------------------------------------------
    # Build the submission summary.
    # ----------------------------------------------

    summary = (
        f"Patient Sex: {sex}\n"
        f"Patient Age: {age}\n"
        f"Injury Status: {injury}\n"
        f"Chief Complain: {complain}\n"
        f"Pain: {pain }\n"
        f"Nurse Assesment: {na}\n"
        f"Body Temprature: {bt}\n"
        f"Mental State: {ms}"
    )

    # ----------------------------------------------
    # Display the submitted information.
    # ----------------------------------------------

    messagebox.showinfo(
        "Form Submitted",
        summary
    )


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
    text="Continue  →",
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