"""Gera um perfil reproduzivel do dataset usado no projeto."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "telco_churn.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "data_profile.json"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisa o dataset IBM Telco Customer Churn."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_value(value: Any) -> Any:
    """Converte escalares do pandas/numpy para tipos serializaveis em JSON."""
    return value.item() if hasattr(value, "item") else value


def build_profile(dataset_path: Path) -> dict[str, Any]:
    dataframe = pd.read_csv(dataset_path)
    cleaned = dataframe.replace(r"^\s*$", pd.NA, regex=True)

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"], errors="coerce"
        )

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Coluna target ausente: {TARGET_COLUMN}")

    feature_frame = cleaned.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    numeric_features = [
        column for column in NUMERIC_FEATURES if column in feature_frame.columns
    ]
    categorical_features = [
        column for column in feature_frame.columns if column not in numeric_features
    ]

    target_counts = {
        str(label): int(count)
        for label, count in cleaned[TARGET_COLUMN].value_counts(dropna=False).items()
    }
    target_percentages = {
        str(label): round(float(percent), 2)
        for label, percent in (
            cleaned[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100
        ).items()
    }
    missing_by_column = {
        column: int(count)
        for column, count in cleaned.isna().sum().items()
        if count > 0
    }
    unique_values = {
        column: int(cleaned[column].nunique(dropna=True))
        for column in cleaned.columns
    }

    return {
        "dataset": {
            "file": dataset_path.name,
            "sha256": file_sha256(dataset_path),
            "rows": int(cleaned.shape[0]),
            "columns": int(cleaned.shape[1]),
            "duplicate_rows": int(cleaned.duplicated().sum()),
            "duplicate_customer_ids": int(cleaned[ID_COLUMN].duplicated().sum()),
        },
        "problem": {
            "type": "binary_classification",
            "target": TARGET_COLUMN,
            "positive_class": "Yes",
            "target_counts": target_counts,
            "target_percentages": target_percentages,
        },
        "data_quality": {
            "missing_by_column_after_blank_normalization": missing_by_column,
        },
        "features": {
            "identifier_to_remove": ID_COLUMN,
            "numeric": numeric_features,
            "categorical": categorical_features,
            "unique_values_by_column": {
                key: python_value(value) for key, value in unique_values.items()
            },
        },
    }


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_path}")

    profile = build_profile(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"\nPerfil salvo em: {output_path}")


if __name__ == "__main__":
    main()
