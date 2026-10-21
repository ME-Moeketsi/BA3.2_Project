import unittest
from unittest.mock import patch

from NLP.conversational_assistant import PredictCareAssistant, urgent_warning


class TestUrgentWarning(unittest.TestCase):
    def test_chest_pain_and_breathing_difficulty(self):
        self.assertIsNotNone(urgent_warning("I have chest pain and difficulty breathing"))

    def test_reverse_order(self):
        self.assertIsNotNone(urgent_warning("I cannot breathe and I have chest pain"))

    def test_non_red_flag_phrase(self):
        self.assertIsNone(urgent_warning("I have had a mild cough for two days"))


class TestAssistant(unittest.TestCase):
    @patch("NLP.conversational_assistant.ollama_chat", return_value="Hello. How can I help?")
    def test_chat_uses_ollama(self, mock_chat):
        assistant = PredictCareAssistant()
        self.assertEqual(assistant.respond("Hello"), "Hello. How can I help?")
        mock_chat.assert_called_once()

    @patch("NLP.conversational_assistant.ollama_chat")
    def test_urgent_message_does_not_call_ollama(self, mock_chat):
        assistant = PredictCareAssistant()
        answer = assistant.respond("I have chest pain and difficulty breathing")
        self.assertIn("medical emergency", answer)
        mock_chat.assert_not_called()

    def test_empty_message(self):
        assistant = PredictCareAssistant()
        self.assertEqual(assistant.respond("  "), "Please type a message first.")

    @patch("NLP.conversational_assistant._load_nlp_predictor", return_value=lambda complaint: {"predicted_ktas": 2, "emergency_category": "Emergency"})
    @patch("NLP.conversational_assistant.ollama_chat", return_value="The classifier returned level 2.")
    def test_prediction_keeps_classifier_result(self, mock_chat, mock_predict):
        assistant = PredictCareAssistant()
        answer = assistant.predict_complaint("headache")
        self.assertIn("KTAS level 2", answer)
        self.assertIn("Plain-language explanation from Ollama", answer)
        mock_predict.assert_called_once()

    @patch("NLP.conversational_assistant._load_nlp_predictor")
    @patch("NLP.conversational_assistant.ollama_chat")
    def test_urgent_complaint_skips_prediction_and_ollama(self, mock_chat, mock_predict):
        assistant = PredictCareAssistant()
        answer = assistant.predict_complaint("chest pain and difficulty breathing")
        self.assertIn("medical emergency", answer)
        mock_predict.assert_not_called()
        mock_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
