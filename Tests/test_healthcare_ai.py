"""
Predict-Care AI
Traditional ML, NLP, and deep-learning integration tests.

Run from the project root:
    python -m unittest Tests.test_healthcare_ai -v

Here I use the project's saved model packages and cleaned datasets. These tests
do not train models or change saved artifacts.
"""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import healthcare_ai as ai
from deep_learning import (
    TRIAGE_FEATURES as DEEP_LEARNING_TRIAGE_FEATURES,
    load_deep_learning_model,
    predict_ktas as predict_deep_learning_ktas,
)
from NLP.nlp_pipeline import predict_ktas as predict_nlp_ktas


PROJECT_DIR = Path(__file__).resolve().parent.parent


def find_dataset(filename):
    for folder in ("data", "Data"):
        path = PROJECT_DIR / folder / filename
        if path.is_file():
            return path
    return None


class HealthcareAIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.triage_package = ai.load_triage_model()
        cls.flow_package = ai.load_patient_flow_model()
        cls.medication_package = ai.load_medication_wait_model()
        cls.combined_package = ai.load_combined_wait_model()
        cls.total_time_package = ai.load_total_time_excluding_med_wait_model()
        (
            cls.deep_learning_model,
            cls.deep_learning_preprocessor,
            cls.deep_learning_feature_names,
        ) = load_deep_learning_model()

        triage_path = find_dataset("ktas_cleaned.csv")
        flow_path = find_dataset("patient_flow_cleaned.csv")
        if triage_path is None or flow_path is None:
            raise FileNotFoundError(
                "The cleaned datasets must be in data/ or Data/ for integration tests."
            )

        triage_df = pd.read_csv(triage_path)
        flow_df = pd.read_csv(flow_path)

        cls.triage_record = (
            triage_df[cls.triage_package["feature_columns"]]
            .dropna()
            .iloc[0]
            .to_dict()
        )
        cls.deep_learning_records = (
            triage_df[DEEP_LEARNING_TRIAGE_FEATURES]
            .dropna()
            .iloc[:2]
            .copy()
        )
        cls.flow_record = (
            flow_df[cls.flow_package["feature_columns"]]
            .dropna()
            .iloc[0]
            .to_dict()
        )

    def test_triage_prediction_is_valid_ktas(self):
        result = ai.predict_triage(self.triage_record, self.triage_package)
        self.assertEqual(len(result), 1)
        self.assertIn(int(result[0]), {1, 2, 3, 4, 5})

    def test_unseen_triage_category_does_not_crash(self):
        record = dict(self.triage_record)
        record["Chief_complain"] = "__UNSEEN_TEST_CATEGORY__"
        result = ai.predict_triage(record, self.triage_package)
        self.assertIn(int(result[0]), {1, 2, 3, 4, 5})

    def test_infinite_triage_value_is_rejected(self):
        record = dict(self.triage_record)
        record["Age"] = np.inf
        with self.assertRaises((ValueError, TypeError)):
            ai.predict_triage(record, self.triage_package)

    def test_all_flow_predictors_return_finite_nonnegative_values(self):
        predictors = [
            (ai.predict_doctor_wait, self.flow_package),
            (ai.predict_medication_wait, self.medication_package),
            (ai.predict_combined_wait, self.combined_package),
            (ai.predict_total_time_excluding_med_wait, self.total_time_package),
        ]
        for predictor, package in predictors:
            with self.subTest(predictor=predictor.__name__):
                result = np.asarray(predictor(self.flow_record, package)).reshape(-1)
                self.assertEqual(result.size, 1)
                self.assertTrue(np.isfinite(result[0]))
                self.assertGreaterEqual(result[0], 0)

    def test_missing_flow_feature_is_rejected(self):
        record = dict(self.flow_record)
        record.pop("ArrivalTime", None)
        with self.assertRaises(ValueError):
            ai.predict_doctor_wait(record, self.flow_package)

    def test_nan_arrival_time_is_rejected(self):
        record = dict(self.flow_record)
        record["ArrivalTime"] = np.nan
        with self.assertRaises(ValueError):
            ai.predict_doctor_wait(record, self.flow_package)

    def test_infinite_arrival_time_is_rejected(self):
        record = dict(self.flow_record)
        record["ArrivalTime"] = np.inf
        with self.assertRaises(ValueError):
            ai.predict_doctor_wait(record, self.flow_package)

    def test_negative_arrival_time_is_rejected(self):
        record = dict(self.flow_record)
        record["ArrivalTime"] = -1
        with self.assertRaises(ValueError):
            ai.predict_doctor_wait(record, self.flow_package)

    def test_invalid_priority_is_rejected(self):
        record = dict(self.flow_record)
        record["Priority"] = 99
        with self.assertRaises(ValueError):
            ai.predict_doctor_wait(record, self.flow_package)

    def test_unknown_scenario_is_rejected(self):
        record = dict(self.flow_record)
        record["Scenario"] = "__UNKNOWN_SCENARIO__"
        with self.assertRaises(ValueError):
            ai.predict_doctor_wait(record, self.flow_package)

    def test_empty_batch_is_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            ai.predict_doctor_wait(pd.DataFrame(), self.flow_package)


class NLPIntegrationTests(unittest.TestCase):
    def test_nlp_prediction_returns_valid_ktas(self):
        result = predict_nlp_ktas("chest pain")
        self.assertIn(int(result["predicted_ktas"]), {1, 2, 3, 4, 5})
        self.assertIn(result["emergency_category"], {"Emergency", "Non-Emergency"})

    def test_nlp_rejects_blank_complaint(self):
        with self.assertRaises((ValueError, TypeError)):
            predict_nlp_ktas("   ")


class DeepLearningIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.preprocessor, cls.feature_names = load_deep_learning_model()
        triage_path = find_dataset("ktas_cleaned.csv")
        if triage_path is None:
            raise FileNotFoundError(
                "The cleaned KTAS dataset must be in data/ or Data/ for integration tests."
            )
        data = pd.read_csv(triage_path)
        missing = [
            column for column in DEEP_LEARNING_TRIAGE_FEATURES
            if column not in data.columns
        ]
        if missing:
            raise ValueError("Dataset is missing features: " + ", ".join(missing))
        cls.records = data[DEEP_LEARNING_TRIAGE_FEATURES].dropna().iloc[:2].copy()
        if cls.records.empty:
            raise ValueError("No usable patient records are available for testing.")

    def test_deep_learning_prediction_returns_valid_ktas(self):
        predictions, probabilities = predict_deep_learning_ktas(
            self.model, self.preprocessor, self.records.iloc[[0]]
        )
        self.assertEqual(np.asarray(predictions).shape, (1,))
        self.assertIn(int(predictions[0]), {1, 2, 3, 4, 5})

        probabilities = np.asarray(probabilities)
        self.assertEqual(probabilities.shape, (1, 5))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all())
        self.assertTrue((probabilities <= 1).all())
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5))

    def test_deep_learning_supports_two_patient_batch(self):
        if len(self.records) < 2:
            self.skipTest("The cleaned dataset contains fewer than two usable records.")
        predictions, probabilities = predict_deep_learning_ktas(
            self.model, self.preprocessor, self.records
        )
        self.assertEqual(np.asarray(predictions).shape, (2,))
        self.assertTrue(
            all(int(value) in {1, 2, 3, 4, 5} for value in predictions)
        )
        self.assertEqual(np.asarray(probabilities).shape, (2, 5))

    def test_deep_learning_rejects_missing_feature(self):
        record = self.records.iloc[[0]].drop(columns=[DEEP_LEARNING_TRIAGE_FEATURES[0]])
        with self.assertRaises(ValueError):
            predict_deep_learning_ktas(self.model, self.preprocessor, record)

    def test_deep_learning_rejects_empty_batch(self):
        empty_batch = self.records.iloc[0:0]
        with self.assertRaises((ValueError, TypeError)):
            predict_deep_learning_ktas(self.model, self.preprocessor, empty_batch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
