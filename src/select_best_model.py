"""Seleciona o melhor run e grava a URI do modelo para a futura API."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.model_selection import choose_best_run, has_selection_metrics
from src.preprocessing import create_train_test_split, load_dataset
from src.train_experiments import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_DATABASE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "best_model.json"
SELECTION_REASON = (
    "Maior F1-score; recall usado como primeiro desempate e accuracy como "
    "segundo desempate."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seleciona o melhor MLP registrado no MLflow."
    )
    parser.add_argument(
        "--tracking-database", type=Path, default=DEFAULT_TRACKING_DATABASE
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def json_parameters(parameters: dict[str, str]) -> dict[str, Any]:
    """Mantem os valores conforme registrados na interface do MLflow."""
    return dict(sorted(parameters.items()))


def main() -> None:
    args = parse_args()
    tracking_database = args.tracking_database.resolve()
    if not tracking_database.is_file():
        raise FileNotFoundError(
            "Banco do MLflow nao encontrado. Execute primeiro: "
            "python -m src.train_experiments"
        )

    tracking_uri = f"sqlite:///{tracking_database.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento nao encontrado: {args.experiment_name}")

    finished_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1000,
    )
    eligible_runs = [run for run in finished_runs if has_selection_metrics(run)]
    best_run = choose_best_run(eligible_runs)

    artifact_paths = {item.path for item in client.list_artifacts(best_run.info.run_id)}
    if "model" not in artifact_paths:
        raise RuntimeError(
            f"O run {best_run.info.run_id} nao possui o artifact model."
        )

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    dataframe = load_dataset()
    split = create_train_test_split(dataframe)
    sample_predictions = model.predict(split.x_test.head(3)).astype(int).tolist()

    # Mantem somente um vencedor marcado, mesmo se o script for executado outra vez.
    for run in eligible_runs:
        client.set_tag(run.info.run_id, "selected_as_best", "false")
    client.set_tag(best_run.info.run_id, "selected_as_best", "true")
    client.set_tag(best_run.info.run_id, "selection_metric", "f1_score")
    client.set_tag(best_run.info.run_id, "selection_reason", SELECTION_REASON)

    run_name = best_run.data.tags.get("mlflow.runName", best_run.info.run_id)
    metrics = {
        name: float(value)
        for name, value in sorted(best_run.data.metrics.items())
    }
    selection = {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": args.experiment_name,
        "run_name": run_name,
        "run_id": best_run.info.run_id,
        "model_uri": model_uri,
        "selection_policy": {
            "primary_metric": "f1_score",
            "first_tiebreaker": "recall",
            "second_tiebreaker": "accuracy",
            "reason": SELECTION_REASON,
        },
        "metrics": metrics,
        "parameters": json_parameters(best_run.data.params),
        "validation": {
            "model_artifact_found": True,
            "model_reload_succeeded": True,
            "sample_predictions": sample_predictions,
        },
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Melhor modelo selecionado com sucesso.")
    print(f"Run: {run_name}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Model URI: {model_uri}")
    print(f"Configuracao salva em: {output_path}")


if __name__ == "__main__":
    main()

