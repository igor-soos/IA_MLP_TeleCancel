"""Testes rapidos da configuracao dos experimentos MLP."""

from __future__ import annotations

import unittest

from sklearn.neural_network import MLPClassifier

from src.training import EXPERIMENT_CONFIGS, build_mlp_pipeline


class TrainingConfigurationTest(unittest.TestCase):
    def test_exactly_five_experiments_are_configured(self) -> None:
        self.assertEqual(len(EXPERIMENT_CONFIGS), 5)
        self.assertEqual(
            [config.name for config in EXPERIMENT_CONFIGS],
            ["MLP-01", "MLP-02", "MLP-03", "MLP-04", "MLP-05"],
        )

    def test_configurations_are_not_all_equal(self) -> None:
        variations = {
            (
                config.hidden_layer_sizes,
                config.activation,
                config.learning_rate_init,
                config.batch_size,
                config.alpha,
            )
            for config in EXPERIMENT_CONFIGS
        }
        self.assertEqual(len(variations), 5)

    def test_pipeline_contains_preprocessor_and_mlp(self) -> None:
        pipeline = build_mlp_pipeline(EXPERIMENT_CONFIGS[0])
        self.assertEqual(list(pipeline.named_steps), ["preprocessor", "mlp"])
        self.assertIsInstance(pipeline.named_steps["mlp"], MLPClassifier)
        self.assertTrue(pipeline.named_steps["mlp"].early_stopping)


if __name__ == "__main__":
    unittest.main()

