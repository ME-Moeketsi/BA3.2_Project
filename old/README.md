# Archived files (not used by the active system)

These files were superseded during the consolidation of the Predict-Care AI
NLP pipeline. They are kept for traceability only and are NOT imported by
any active module.

| File | Replaced by | Reason |
|------|-------------|--------|
| predict_ktas.py | pipeline.py (predict_ktas) | Weaker validation; tuple return replaced by dict. |
| nlp_pipeline.py | pipeline.py | Renamed; extended with predict_waiting_time + route. |
| ktas_input_menu.py | main_menu.py | Merged with waiting_time_input.py into one CLI. |
| ktas_speech_input.py | main_menu.py (get_speech_input) | Speech flow merged into the unified menu. |
| waiting_time_input.py | main_menu.py + pipeline.py | Input collection → menu; prediction hook → pipeline. |
| compare_models.py | — | One-off model-selection experiment; not part of the system. |
| conversational_assistant.py | local_language_engine.py | Only one chat front-end is kept (Ollama-free). |