"""Testes dos dados de entrada e da resposta da API."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from src.api import CustomerInput, PredictionService


VALID_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.70,
    "TotalCharges": 190.50,
}


class FakeModel:
    def predict(self, dataframe):
        return np.array([1])

    def predict_proba(self, dataframe):
        return np.array([[0.18, 0.82]])


class ApiTest(unittest.TestCase):
    def test_valid_customer_is_accepted(self) -> None:
        customer = CustomerInput(**VALID_CUSTOMER)
        self.assertEqual(customer.Contract, "Month-to-month")
        self.assertEqual(customer.tenure, 2)

    def test_invalid_category_is_rejected(self) -> None:
        invalid_customer = {**VALID_CUSTOMER, "Contract": "Weekly"}
        with self.assertRaises(ValidationError):
            CustomerInput(**invalid_customer)

    def test_negative_charge_is_rejected(self) -> None:
        invalid_customer = {**VALID_CUSTOMER, "MonthlyCharges": -1}
        with self.assertRaises(ValidationError):
            CustomerInput(**invalid_customer)

    def test_service_formats_prediction_response(self) -> None:
        service = PredictionService()
        service.model = FakeModel()
        service.metadata = {"run_name": "MLP-05"}
        response = service.predict(CustomerInput(**VALID_CUSTOMER))
        self.assertEqual(response.prediction, 1)
        self.assertEqual(response.label, "Churn")
        self.assertEqual(response.probability, 0.82)
        self.assertEqual(response.model_run, "MLP-05")


if __name__ == "__main__":
    unittest.main()

