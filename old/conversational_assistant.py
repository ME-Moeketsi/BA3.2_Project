"""Predict-Care AI local conversational assistant.

This module connects a local Ollama model to the existing NLP KTAS classifier.
It does not train models, change the GUI, or replace clinical assessment.
Run from the project root with: python -m NLP.conversational_assistant
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = """You are the conversational assistant for Predict-Care AI, an academic healthcare decision-support prototype.

Rules:
- Speak clearly, respectfully, and in plain language.
- The project uses a five-level KTAS classification, where level 1 is the most urgent and level 5 the least urgent in the project's convention.
- You may explain general concepts, ask for clarification, and explain a prediction that the application has explicitly supplied to you.
- Never invent a patient's KTAS level, clinical diagnosis, probability, or waiting-time estimate.
- A language-model answer is not a clinical assessment. Do not tell users that they are safe or that they can delay care based on this conversation.
- If someone reports potentially life-threatening symptoms, tell them to seek emergency medical help immediately. Do not delay that advice to ask follow-up questions.
- Do not claim that the application is clinically validated.
- For non-emergency situations, encourage professional medical assessment when appropriate.
"""

# These are conservative phrase checks, not a diagnostic or complete triage system.
_URGENT_PATTERNS = [
    re.compile(r"\b(chest pain|chest pressure|crushing chest|tightness in (my )?chest)\b.*\b(short(ness)? of breath|difficulty breathing|trouble breathing|can't breathe|cannot breathe|gasping)\b", re.I),
    re.compile(r"\b(short(ness)? of breath|difficulty breathing|trouble breathing|can't breathe|cannot breathe|gasping)\b.*\b(chest pain|chest pressure|crushing chest|tightness in (my )?chest)\b", re.I),
    re.compile(r"\b(unconscious|not breathing|stopped breathing|severe bleeding|bleeding heavily|stroke symptoms|face drooping|sudden weakness on one side|suicide attempt|overdose)\b", re.I),
    re.compile(r"\b(sudden|severe)\b.{0,35}\b(breathing difficulty|difficulty breathing|shortness of breath|chest pain|confusion|weakness)\b", re.I),
]


def urgent_warning(text: str) -> str | None:
    """Return a conservative emergency warning for recognized red-flag phrases."""
    if any(pattern.search(text) for pattern in _URGENT_PATTERNS):
        return (
            "This may be a medical emergency. Please seek emergency medical help now "
            "or contact your local emergency number. Do not wait for an AI prediction "
            "or continue this chat to decide whether it is serious. Predict-Care AI is "
            "an academic prototype and cannot assess your condition safely."
        )
    return None


def ollama_chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
    endpoint: str = OLLAMA_CHAT_URL,
) -> str:
    """Send a chat request to the local Ollama HTTP API and return its text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            "Could not connect to Ollama at http://127.0.0.1:11434. "
            "Make sure the Ollama application is running."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned a response that was not valid JSON.") from exc

    answer = result.get("message", {}).get("content", "")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Ollama returned an empty response.")
    return answer.strip()


class PredictCareAssistant:
    """Manage local chat history and call the existing NLP classifier on request."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def set_model(self, model: str) -> None:
        if not model.strip():
            raise ValueError("Model name cannot be empty.")
        self.model = model.strip()

    def respond(self, user_text: str) -> str:
        """Respond conversationally, with a deterministic red-flag safety check."""
        text = user_text.strip()
        if not text:
            return "Please type a message first."

        warning = urgent_warning(text)
        if warning:
            # Here I put the urgent safety instruction before any model-generated text.
            return warning

        self.history.append({"role": "user", "content": text})
        # Here I keep only recent exchanges so the local prompt does not grow forever.
        system_message = self.history[0]
        recent_messages = self.history[1:][-MAX_HISTORY_MESSAGES:]
        answer = ollama_chat([system_message, *recent_messages], model=self.model)
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def predict_complaint(self, complaint: str) -> str:
        """Call the existing trained NLP classifier and explain its returned output."""
        text = complaint.strip()
        if not text:
            return "Please provide a chief complaint after /predict."

        warning = urgent_warning(text)
        if warning:
            # Here I do not let an experimental classifier delay emergency advice.
            return warning

        try:
            # Here I load the project's existing classifier only when it is requested.
            predict_ktas = _load_nlp_predictor()
            result = predict_ktas(text)
        except (FileNotFoundError, ImportError, ValueError, OSError) as exc:
            return (
                "I could not run the existing NLP prediction model. "
                f"Check that the trained model files and Python environment are ready. Details: {exc}"
            )
        except Exception as exc:  # Keep the CLI alive if a model/library error occurs.
            return f"The NLP prediction failed ({type(exc).__name__}: {exc}). No KTAS result was produced."

        level = result.get("predicted_ktas")
        category = result.get("emergency_category", "Unspecified")
        if level not in (1, 2, 3, 4, 5):
            return (
                "The classifier returned an unexpected KTAS value. "
                "I will not interpret it as a valid result. Please check the trained model."
            )

        grounded_result = (
            f"The existing NLP classifier returned KTAS level {level} "
            f"and category '{category}' for the complaint: {text!r}. "
            "This is an experimental model output, not a clinical assessment. "
            "Do not use it to decide that symptoms are safe or to delay care."
        )
        try:
            explanation = ollama_chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT + "\nExplain only the supplied result. Do not alter the level or category, infer a diagnosis, or claim certainty."},
                    {"role": "user", "content": grounded_result},
                ],
                model=self.model,
            )
        except (ConnectionError, RuntimeError, TimeoutError) as exc:
            return grounded_result + f"\n\nThe language-model explanation is unavailable: {exc}"

        # Here I include the exact classifier result separately from the generated explanation.
        return grounded_result + "\n\nPlain-language explanation from Ollama:\n" + explanation


def _load_nlp_predictor():
    """Load the existing project classifier lazily to avoid startup side effects."""
    from old.nlp_pipeline import predict_ktas
    return predict_ktas


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /help                 Show these commands\n"
        "  /model                Show the current Ollama model\n"
        "  /model MODEL_NAME     Switch model (e.g. /model phi3:mini)\n"
        "  /predict COMPLAINT    Run the existing NLP KTAS classifier\n"
        "  /reset                Clear conversation history\n"
        "  /quit                 Exit the assistant\n\n"
        "Type any other message to chat with Ollama.\n"
        "For urgent symptoms, seek emergency help rather than relying on this prototype.\n"
    )


def main() -> None:
    assistant = PredictCareAssistant()
    print("Predict-Care AI — Local Language Assistant")
    print(f"Ollama model: {assistant.model}")
    print("This is an academic prototype, not a medical service. Type /help for commands.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Session ended.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/quit", "/exit"}:
            print("Assistant: Goodbye.")
            break
        if user_text.lower() == "/help":
            print_help()
            continue
        if user_text.lower() == "/model":
            print(f"Assistant: Current model is {assistant.model}")
            continue
        if user_text.lower().startswith("/model "):
            try:
                assistant.set_model(user_text.split(maxsplit=1)[1])
                assistant.reset()
                print(f"Assistant: Switched to {assistant.model}; conversation history cleared.")
            except (IndexError, ValueError) as exc:
                print(f"Assistant: {exc}")
            continue
        if user_text.lower() == "/reset":
            assistant.reset()
            print("Assistant: Conversation history cleared.")
            continue
        if user_text.lower().startswith("/predict"):
            complaint = user_text[len("/predict"):].strip()
            if complaint.lower().startswith("complaint:"):
                complaint = complaint.split(":", 1)[1].strip()
            print("Assistant:", assistant.predict_complaint(complaint))
            continue

        try:
            print("Assistant:", assistant.respond(user_text))
        except (ConnectionError, RuntimeError, TimeoutError) as exc:
            print(f"Assistant: {exc}")


if __name__ == "__main__":
    main()
