"""Carregamento, divisao e pre-processamento dos dados de churn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "telco_churn.csv"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
TARGET_MAPPING = {"No": 0, "Yes": 1}

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


@dataclass(frozen=True)
class DataSplit:
    """Dados separados sem aplicar transformacoes antes da hora."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(dataset_path: Path | str = DEFAULT_DATASET) -> pd.DataFrame:
    """Carrega o CSV e normaliza os campos que precisam de limpeza inicial."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset nao encontrado: {path.resolve()}")

    dataframe = pd.read_csv(path)
    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Colunas obrigatorias ausentes: " + ", ".join(sorted(missing_columns))
        )

    # O CSV possui espacos em branco em TotalCharges. Eles viram NaN para que
    # o imputador do pipeline possa trata-los usando somente os dados de treino.
    dataframe = dataframe.replace(r"^\s*$", np.nan, regex=True)
    dataframe["TotalCharges"] = pd.to_numeric(
        dataframe["TotalCharges"], errors="coerce"
    )

    invalid_targets = set(dataframe[TARGET_COLUMN].dropna().unique()) - set(
        TARGET_MAPPING
    )
    if invalid_targets:
        raise ValueError(f"Valores inesperados no target: {sorted(invalid_targets)}")
    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("O target Churn possui valores ausentes.")

    return dataframe


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove o identificador e converte Churn de No/Yes para 0/1."""
    x = dataframe.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
    y = dataframe[TARGET_COLUMN].map(TARGET_MAPPING).astype("int64")
    return x, y


def create_train_test_split(
    dataframe: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Separa treino e teste preservando a proporcao das classes."""
    x, y = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DataSplit(x_train, x_test, y_train, y_test)


def build_preprocessor() -> ColumnTransformer:
    """Cria as transformacoes que depois farao parte do pipeline do MLP."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

