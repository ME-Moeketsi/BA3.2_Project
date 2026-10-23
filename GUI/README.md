# Predict-Care AI desktop GUI — connected prediction models

## Start (Windows 11, PowerShell or VS Code terminal)

From the extracted `BA3.2_Project-main` folder:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python GUI/GUI.py
```

On Linux, install your distribution's Tkinter package first (such as
`sudo apt install python3-tk`), then run `python GUI/GUI.py` from the project
root. No network API or external LLM is needed for typed predictions.

**The updated ZIP includes the trained files**:

- `Models/ktas_nlp_model.pkl` and `Models/ktas_tfidf_vectorizer.pkl`:
  complaint-text KTAS model and text vectorizer.
- `Models/waiting_time_model.pkl`: doctor waiting time and total in-hospital
  time regressors, plus their feature metadata.

The original upload had the scripts and datasets but not these model binaries.
They were trained on the bundled datasets. To retrain them:

```powershell
python NLP/train_ktas_model.py
python NLP/train_waiting_time_model.py
```

**Important:** `.pkl` files must be loaded only from trusted sources; do not
replace them with model files from strangers.

## Using the GUI

1. Type a **chief complaint** in the patient's form. Other patient fields are
   optional; the **NLP classifier uses the complaint text only**. Do not assume
   the GUI combines age, vitals and other clinical observations into that model.
2. To obtain **waiting times**, enter **Arrival time (simulation minutes)**,
   number of available doctors (1–4), and peak arrivals (Yes/No). The arrival
   time is the simulation's elapsed-minute feature (observed 3–1107), **not**
   an actual time of day or number of minutes the patient has waited.
3. Under **Ask AI**, type a question such as `predict both waiting times`,
   `how long until I see a doctor`, or `total time in hospital`.
4. Click **Calculate predictions**. The right-hand panels display the predicted
   KTAS level, an experimental urgency indicator derived from KTAS, waiting
   estimates (where enough data is supplied), and the relevant safety notes.
5. **Preview patient summary** updates the summary only. Voice assessment is
   optional and requires `pip install -r requirements-optional.txt` and a
   compatible microphone/speech setup. The voice feature may use Google's
   speech recognition service and requires an internet connection.

The pipeline maps KTAS 1/2 to simulation priority 3, KTAS 3 to 2, and KTAS
4/5 to 1. This is the **project's operational mapping**, not an established
clinical risk classification. Inputs outside the simulation training domain
are rejected rather than silently guessed. Unknown or ambiguous waiting
requests prompt for clarification without inventing a result.

## Limitations

- KTAS text predictions and priority labels are **educational prototypes**, not
  diagnoses, real-world patient health-risk probabilities, or clinical triage.
- The waiting-time model was trained on a simulated hospital dataset. Its
  estimates are **not validated actual wait times at any hospital**.
- The preexisting deep learning `.keras` file cannot currently be used for
  inference: the matching fitted
  `Models/deep_learning_triage_preprocessor.pkl` was missing from the uploaded
  archive. This GUI intentionally does **not** try to load the neural network
  without its required preprocessor or substitute made-up input values.
- Emergency warnings supersede wait estimates. Call local emergency services
  or consult a qualified clinician for concerning symptoms; do not delay care
  based on this program.

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named sklearn`, `joblib`, or `pandas` | Install `requirements.txt` using the same Python environment as the GUI |
| `NLP model not found` / `Waiting-time model not found` | Restore the files under `Models/` or run the two training commands above |
| `No module named tkinter` | Install Python with Tcl/Tk or your OS Tk package |
| Voice packages missing | The typed GUI still works; optionally install `requirements-optional.txt` |
| No graphical display (`$DISPLAY`) | Run locally on a desktop or use a configured graphical environment |
