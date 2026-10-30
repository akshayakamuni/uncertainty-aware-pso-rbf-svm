import unittest

import pandas as pd

from analyze_final_results import (
    pair_against_lambda_zero,
    seed_dataset_preprocessing_summary,
    validate_result_domain,
)


class FinalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "seeds": [42],
            "datasets": ["sample"],
            "preprocessing_configs": [{"name": "median"}],
            "outer_folds": 2,
            "lambda_values": [0.0, 0.25],
        }
        self.frame = pd.DataFrame(
            [
                {
                    "seed": 42,
                    "dataset": "sample",
                    "preprocessing_config": "median",
                    "outer_fold": fold,
                    "lambda": lambda_value,
                    "outer_roc_auc": baseline + delta,
                    "balanced_accuracy": baseline + delta,
                    "sensitivity": baseline + delta,
                    "specificity": baseline + delta,
                    "f1_score": baseline + delta,
                    "inner_to_outer_loss_gap": baseline + delta,
                    "absolute_inner_to_outer_loss_gap": baseline + delta,
                    "inner_mean_loss": baseline + delta,
                    "inner_std_loss": baseline + delta,
                }
                for fold, baseline in ((1, 0.5), (2, 0.6))
                for lambda_value, delta in ((0.0, 0.0), (0.25, 0.1))
            ]
        )

    def test_accepts_complete_expected_key_domain(self):
        validate_result_domain(self.frame, self.config, "fixture")

    def test_rejects_missing_fold_or_lambda_key(self):
        with self.assertRaisesRegex(ValueError, "complete configured key domain"):
            validate_result_domain(self.frame.iloc[:-1], self.config, "fixture")

    def test_rejects_duplicate_result_key(self):
        duplicate = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate result keys"):
            validate_result_domain(duplicate, self.config, "fixture")

    def test_pairs_each_lambda_against_matched_zero(self):
        paired = pair_against_lambda_zero(self.frame, self.config["lambda_values"])
        self.assertEqual(len(paired), 2)
        self.assertTrue(
            (paired["delta_outer_roc_auc"].sub(0.1).abs() < 1e-12).all()
        )

    def test_summary_aggregates_folds_within_seed_and_configuration(self):
        paired = pair_against_lambda_zero(self.frame, self.config["lambda_values"])
        summary = seed_dataset_preprocessing_summary(paired)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["seed"], 42)
        self.assertAlmostEqual(summary.iloc[0]["delta_outer_roc_auc"], 0.1)

    def test_pairing_rejects_missing_lambda(self):
        incomplete = self.frame[self.frame["lambda"] == 0.0]
        with self.assertRaisesRegex(ValueError, "Missing lambda"):
            pair_against_lambda_zero(incomplete, self.config["lambda_values"])


if __name__ == "__main__":
    unittest.main()
