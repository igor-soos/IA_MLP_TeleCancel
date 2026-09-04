"""API local para predicao de churn com o modelo selecionado no MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "config" / "best_model.json"
DEFAULT_TRACKING_DATABASE = PROJECT_ROOT / "mlflow.db"

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    """Dados originais de um cliente, antes do pre-processamento."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    label: Literal["Churn", "No churn"]
    probability: float
    model_run: str


class PredictionService:
    """Carrega uma vez o pipeline escolhido e o reutiliza nas requisicoes."""

    def __init__(
        self,
        tracking_database: Path = DEFAULT_TRACKING_DATABASE,
        model_config: Path = DEFAULT_MODEL_CONFIG,
    ) -> None:
        self.tracking_database = tracking_database
        self.model_config = model_config
        self.model = None
        self.metadata: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        if not self.tracking_database.is_file():
            raise FileNotFoundError(
                "mlflow.db nao encontrado. Execute o treinamento primeiro."
            )
        if not self.model_config.is_file():
            raise FileNotFoundError(
                "config/best_model.json nao encontrado. Execute "
                "python -m src.select_best_model primeiro."
            )

        metadata = json.loads(self.model_config.read_text(encoding="utf-8"))
        model_uri = metadata.get("model_uri")
        if not model_uri:
            raise ValueError("model_uri ausente em config/best_model.json")

        tracking_uri = f"sqlite:///{self.tracking_database.resolve().as_posix()}"
        mlflow.set_tracking_uri(tracking_uri)
        self.model = mlflow.sklearn.load_model(model_uri)
        self.metadata = metadata

    def predict(self, customer: CustomerInput) -> PredictionResponse:
        self.load()
        input_data = customer.model_dump()
        if input_data["TotalCharges"] is None:
            input_data["TotalCharges"] = np.nan
        dataframe = pd.DataFrame([input_data])

        prediction = int(self.model.predict(dataframe)[0])
        probability = float(self.model.predict_proba(dataframe)[0, 1])
        return PredictionResponse(
            prediction=prediction,
            label="Churn" if prediction == 1 else "No churn",
            probability=round(probability, 4),
            model_run=self.metadata.get("run_name", "unknown"),
        )


app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "API local que carrega do MLflow o pipeline MLP selecionado e preve "
        "o risco de cancelamento de um cliente."
    ),
    version="1.0.0",
)
model_service = PredictionService()


def get_model_service() -> PredictionService:
    try:
        model_service.load()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo indisponivel: {exc}",
        ) from exc
    return model_service


@app.get("/", tags=["Status"])
def root() -> dict[str, str]:
    return {
        "message": "Telco Churn Prediction API",
        "documentation": "/docs",
        "health": "/health",
        "prediction": "POST /predict",
    }


@app.get("/health", tags=["Status"])
def health(
    service: PredictionService = Depends(get_model_service),
) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": service.is_loaded,
        "model_run": service.metadata.get("run_name", "unknown"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    customer: CustomerInput,
    service: PredictionService = Depends(get_model_service),
) -> PredictionResponse:
    try:
        return service.predict(customer)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Nao foi possivel realizar a predicao.",
        ) from exc

