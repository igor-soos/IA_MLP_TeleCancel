"""Treina cinco MLPs e registra cada experimento no MLflow."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.preprocessing import (
    DEFAULT_DATASET,
    create_train_test_split,
    load_dataset,
)
from src.training import (
    EXPERIMENT_CONFIGS,
    ExperimentConfig,
    build_mlp_pipeline,
    calculate_classification_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_DATABASE = PROJECT_ROOT / "mlflow.db"
DEFAULT_ARTIFACT_DIRECTORY = PROJECT_ROOT / "mlartifacts"
DEFAULT_SUMMARY = PROJECT_ROOT / "docs" / "experiment_validation.json"
DEFAULT_EXPERIMENT_NAME = "telco-churn-mlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e registra cinco configuracoes de MLP no MLflow."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--tracking-database", type=Path, default=DEFAULT_TRACKING_DATABASE
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def mlflow_parameters(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "hidden_layers": str(config.hidden_layer_sizes),
        "activation": config.activation,
        "learning_rate": config.learning_rate_init,
        "batch_size": config.batch_size,
        "max_iter": config.max_iter,
        "alpha": config.alpha,
        "solver": "adam",
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": 42,
        "test_size": 0.20,
    }


def save_run_artifacts(
    directory: Path,
    run_name: str,
    y_true,
    y_prediction,
    loss_curve: list[float],
    metrics: dict[str, float],
) -> None:
    report = classification_report(
        y_true,
        y_prediction,
        target_names=["No churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    (directory / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_prediction,
        display_labels=["No churn", "Churn"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"Matriz de confusao — {run_name}")
    display.figure_.tight_layout()
    display.figure_.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(1, len(loss_curve) + 1), loss_curve)
    axis.set_title(f"Curva de perda — {run_name}")
    axis.set_xlabel("Iteracao")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "loss_curve.png", dpi=150)
    plt.close(figure)


def train_one_experiment(config: ExperimentConfig, split) -> dict[str, Any]:
    pipeline = build_mlp_pipeline(config)

    with mlflow.start_run(run_name=config.name) as active_run:
        mlflow.set_tags(
            {
                "problem_type": "binary_classification",
                "dataset": "IBM Telco Customer Churn",
                "target": "Churn",
                "positive_class": "Yes",
            }
        )
        mlflow.log_params(mlflow_parameters(config))

        started_at = time.perf_counter()
        pipeline.fit(split.x_train, split.y_train)
        fit_time = time.perf_counter() - started_at

        predictions = pipeline.predict(split.x_test)
        probabilities = pipeline.predict_proba(split.x_test)[:, 1]
        metrics = calculate_classification_metrics(
            split.y_test, predictions, probabilities
        )

        mlp = pipeline.named_steps["mlp"]
        training_metrics = {
            **metrics,
            "fit_time_seconds": float(fit_time),
            "iterations": float(mlp.n_iter_),
            "final_loss": float(mlp.loss_),
            "best_validation_score": float(mlp.best_validation_score_),
        }
        mlflow.log_metrics(training_metrics)

        input_example = split.x_train.head(5).copy()
        # Tipos float na assinatura aceitam os inteiros usuais e tambem futuros
        # valores ausentes sem conflito de schema no MLflow.
        input_example["tenure"] = input_example["tenure"].astype("float64")
        input_example["SeniorCitizen"] = input_example["SeniorCitizen"].astype(
            "float64"
        )
        signature = infer_signature(input_example, pipeline.predict(input_example))

        with tempfile.TemporaryDirectory(prefix=f"{config.name.lower()}-") as temp_dir:
            temporary_directory = Path(temp_dir)
            evaluation_directory = temporary_directory / "evaluation"
            evaluation_directory.mkdir()
            save_run_artifacts(
                evaluation_directory,
                config.name,
                split.y_test,
                predictions,
                mlp.loss_curve_,
                training_metrics,
            )
            mlflow.log_artifacts(evaluation_directory, artifact_path="evaluation")

            # O modelo e salvo e depois enviado como artifact tradicional. Isso
            # faz a pasta "model" aparecer dentro do run na interface do MLflow.
            model_directory = temporary_directory / "model"
            mlflow.sklearn.save_model(
                sk_model=pipeline,
                path=model_directory,
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
                pip_requirements=[
                    "mlflow==3.15.2",
                    "pandas==2.3.3",
                    "scikit-learn==1.9.0",
                ],
            )
            mlflow.log_artifacts(model_directory, artifact_path="model")

        run_id = active_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        reloaded_model = mlflow.sklearn.load_model(model_uri)
        reload_ok = bool(
            np.array_equal(
                pipeline.predict(input_example),
                reloaded_model.predict(input_example),
            )
        )
        mlflow.set_tag("model_reload_validated", str(reload_ok).lower())
        if not reload_ok:
            raise RuntimeError(f"Falha ao recarregar o modelo do run {run_id}")

        return {
            "run_name": config.name,
            "run_id": run_id,
            "model_uri": model_uri,
            "parameters": mlflow_parameters(config),
            "metrics": training_metrics,
            "model_reload_validated": reload_ok,
        }


def main() -> None:
    args = parse_args()
    tracking_database = args.tracking_database.resolve()
    artifact_directory = args.artifact_directory.resolve()
    tracking_database.parent.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)

    # Uma URI SQLite absoluta funciona tanto no Windows quanto no Linux.
    tracking_uri = f"sqlite:///{tracking_database.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            args.experiment_name,
            artifact_location=artifact_directory.as_uri(),
        )
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_id=experiment.experiment_id)

    dataframe = load_dataset(args.dataset)
    split = create_train_test_split(dataframe)

    results = []
    print(f"Experimento MLflow: {args.experiment_name}")
    print(f"Banco local: {tracking_database}")
    print(f"Artifacts locais: {artifact_directory}")

    for position, config in enumerate(EXPERIMENT_CONFIGS, start=1):
        print(f"\n[{position}/{len(EXPERIMENT_CONFIGS)}] Treinando {config.name}...")
        result = train_one_experiment(config, split)
        results.append(result)
        print(
            f"{config.name}: F1={result['metrics']['f1_score']:.4f} | "
            f"Recall={result['metrics']['recall']:.4f} | "
            f"Accuracy={result['metrics']['accuracy']:.4f}"
        )

    summary = {
        "experiment_name": args.experiment_name,
        "tracking_database": str(tracking_database),
        "artifact_directory": str(artifact_directory),
        "run_count": len(results),
        "comparison_note": (
            "Este arquivo valida a execucao. A comparacao oficial deve ser feita "
            "na interface do MLflow."
        ),
        "runs": results,
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nCinco runs registrados com sucesso.")
    print(f"Resumo tecnico salvo em: {summary_path}")
    print("Abra a interface do MLflow para comparar os resultados.")


if __name__ == "__main__":
    main()
