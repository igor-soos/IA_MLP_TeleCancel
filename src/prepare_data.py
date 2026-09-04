"""Executa e documenta o pre-processamento da Etapa 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing import (
    DEFAULT_DATASET,
    build_preprocessor,
    create_train_test_split,
    load_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "preprocessing_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara os dados de churn para o futuro treinamento do MLP."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def class_distribution(target: Any) -> dict[str, Any]:
    counts = target.value_counts().sort_index()
    percentages = target.value_counts(normalize=True).sort_index() * 100
    return {
        "counts": {str(label): int(value) for label, value in counts.items()},
        "percentages": {
            str(label): round(float(value), 2)
            for label, value in percentages.items()
        },
    }


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.dataset)
    split = create_train_test_split(dataframe)
    preprocessor = build_preprocessor()

    # fit_transform ocorre somente no treino para evitar vazamento de dados.
    x_train_processed = preprocessor.fit_transform(split.x_train)
    x_test_processed = preprocessor.transform(split.x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    summary = {
        "split": {
            "random_state": 42,
            "test_size": 0.20,
            "train_rows": int(split.x_train.shape[0]),
            "test_rows": int(split.x_test.shape[0]),
            "raw_feature_count": int(split.x_train.shape[1]),
            "processed_feature_count": int(x_train_processed.shape[1]),
        },
        "target": {
            "mapping": {"No": 0, "Yes": 1},
            "train": class_distribution(split.y_train),
            "test": class_distribution(split.y_test),
        },
        "missing_values": {
            "train_before_preprocessing": int(split.x_train.isna().sum().sum()),
            "test_before_preprocessing": int(split.x_test.isna().sum().sum()),
            "train_after_preprocessing": int(np.isnan(x_train_processed).sum()),
            "test_after_preprocessing": int(np.isnan(x_test_processed).sum()),
        },
        "processed_features": feature_names,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Etapa 2 executada com sucesso.")
    print(f"Treino: {summary['split']['train_rows']} clientes")
    print(f"Teste: {summary['split']['test_rows']} clientes")
    print(
        "Features: "
        f"{summary['split']['raw_feature_count']} originais -> "
        f"{summary['split']['processed_feature_count']} apos o pre-processamento"
    )
    print(
        "Valores ausentes apos o pre-processamento: "
        f"{summary['missing_values']['train_after_preprocessing']} no treino e "
        f"{summary['missing_values']['test_after_preprocessing']} no teste"
    )
    print(f"Resumo salvo em: {output_path}")


if __name__ == "__main__":
    main()

