"""Regras reutilizaveis para selecionar o melhor run do MLflow."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


REQUIRED_SELECTION_METRICS = ("f1_score", "recall", "accuracy")


def has_selection_metrics(run: Any) -> bool:
    """Confirma que o run possui todas as metricas usadas na selecao."""
    return all(metric in run.data.metrics for metric in REQUIRED_SELECTION_METRICS)


def selection_key(run: Any) -> tuple[float, float, float]:
    """Prioriza F1, depois recall e, por ultimo, accuracy."""
    metrics = run.data.metrics
    return (
        float(metrics["f1_score"]),
        float(metrics["recall"]),
        float(metrics["accuracy"]),
    )


def choose_best_run(runs: Iterable[Any]) -> Any:
    """Retorna o melhor run segundo a politica definida para churn."""
    eligible_runs = [run for run in runs if has_selection_metrics(run)]
    if not eligible_runs:
        raise ValueError(
            "Nenhum run finalizado possui f1_score, recall e accuracy."
        )
    return max(eligible_runs, key=selection_key)

