"""Configuracoes e funcoes reutilizaveis para treinar os MLPs."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    learning_rate_init: float
    batch_size: int
    max_iter: int
    alpha: float


EXPERIMENT_CONFIGS = [
    ExperimentConfig("MLP-01", (32,), "relu", 0.001, 32, 300, 0.0001),
    ExperimentConfig("MLP-02", (64,), "relu", 0.001, 32, 300, 0.0001),
    ExperimentConfig("MLP-03", (64, 32), "relu", 0.001, 32, 400, 0.0001),
    ExperimentConfig("MLP-04", (64, 32), "tanh", 0.001, 32, 400, 0.0001),
    ExperimentConfig("MLP-05", (128, 64), "relu", 0.0001, 64, 500, 0.001),
]


def build_mlp_pipeline(config: ExperimentConfig) -> Pipeline:
    """Une o pre-processamento e o MLP no mesmo objeto."""
    classifier = MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        solver="adam",
        learning_rate_init=config.learning_rate_init,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        alpha=config.alpha,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("mlp", classifier),
        ]
    )


def calculate_classification_metrics(
    y_true, y_prediction, y_probability
) -> dict[str, float]:
    """Calcula as metricas exigidas para a classificacao de churn."""
    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "precision": float(
            precision_score(y_true, y_prediction, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_prediction, pos_label=1, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_prediction, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
    }

