"""Testes da regra usada para selecionar o melhor MLP."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.model_selection import choose_best_run, has_selection_metrics


@dataclass
class FakeRunData:
    metrics: dict[str, float]


@dataclass
class FakeRun:
    name: str
    data: FakeRunData


def fake_run(name: str, f1: float, recall: float, accuracy: float) -> FakeRun:
    return FakeRun(
        name,
        FakeRunData(
            {"f1_score": f1, "recall": recall, "accuracy": accuracy}
        ),
    )


class ModelSelectionTest(unittest.TestCase):
    def test_highest_f1_is_selected(self) -> None:
        runs = [
            fake_run("MLP-01", 0.57, 0.54, 0.78),
            fake_run("MLP-05", 0.59, 0.56, 0.79),
            fake_run("MLP-02", 0.55, 0.47, 0.80),
        ]
        self.assertEqual(choose_best_run(runs).name, "MLP-05")

    def test_recall_breaks_an_f1_tie(self) -> None:
        runs = [
            fake_run("A", 0.59, 0.50, 0.81),
            fake_run("B", 0.59, 0.56, 0.79),
        ]
        self.assertEqual(choose_best_run(runs).name, "B")

    def test_run_without_required_metric_is_not_eligible(self) -> None:
        incomplete = FakeRun("incomplete", FakeRunData({"f1_score": 0.90}))
        self.assertFalse(has_selection_metrics(incomplete))
        with self.assertRaises(ValueError):
            choose_best_run([incomplete])


if __name__ == "__main__":
    unittest.main()

