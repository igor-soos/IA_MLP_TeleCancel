"""Testes da preparacao de dados da Etapa 2."""

from __future__ import annotations

import unittest

import numpy as np

from src.preprocessing import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_preprocessor,
    create_train_test_split,
    load_dataset,
)


class PreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_dataset()
        cls.split = create_train_test_split(cls.dataframe)
        cls.preprocessor = build_preprocessor()
        cls.x_train_processed = cls.preprocessor.fit_transform(cls.split.x_train)
        cls.x_test_processed = cls.preprocessor.transform(cls.split.x_test)

    def test_split_has_expected_sizes(self) -> None:
        self.assertEqual(len(self.split.x_train), 5634)
        self.assertEqual(len(self.split.x_test), 1409)
        self.assertEqual(len(self.split.y_train), 5634)
        self.assertEqual(len(self.split.y_test), 1409)

    def test_identifier_and_target_are_not_features(self) -> None:
        self.assertNotIn(ID_COLUMN, self.split.x_train.columns)
        self.assertNotIn(TARGET_COLUMN, self.split.x_train.columns)

    def test_target_is_binary_and_stratified(self) -> None:
        self.assertEqual(set(self.split.y_train.unique()), {0, 1})
        overall_rate = (self.dataframe[TARGET_COLUMN] == "Yes").mean()
        train_rate = self.split.y_train.mean()
        test_rate = self.split.y_test.mean()
        self.assertAlmostEqual(train_rate, overall_rate, places=3)
        self.assertAlmostEqual(test_rate, overall_rate, places=3)

    def test_preprocessing_removes_missing_values(self) -> None:
        self.assertGreater(self.split.x_train.isna().sum().sum(), 0)
        self.assertEqual(int(np.isnan(self.x_train_processed).sum()), 0)
        self.assertEqual(int(np.isnan(self.x_test_processed).sum()), 0)

    def test_preprocessing_creates_expected_number_of_features(self) -> None:
        self.assertEqual(self.split.x_train.shape[1], 19)
        self.assertEqual(self.x_train_processed.shape[1], 46)
        self.assertEqual(self.x_test_processed.shape[1], 46)

    def test_unknown_category_can_be_transformed(self) -> None:
        new_data = self.split.x_test.iloc[[0]].copy()
        new_data.loc[:, "PaymentMethod"] = "New payment method"
        transformed = self.preprocessor.transform(new_data)
        self.assertEqual(transformed.shape, (1, 46))


if __name__ == "__main__":
    unittest.main()

