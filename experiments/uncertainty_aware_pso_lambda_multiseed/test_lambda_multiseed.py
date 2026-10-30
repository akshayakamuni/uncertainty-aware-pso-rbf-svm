"""Focused tests for lambda-sensitivity study safeguards and criteria."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_lambda_multiseed as runner
import validate_lambda_multiseed as validator


class LambdaMultiseedTests(unittest.TestCase):
    def test_config_uses_exact_lambda_list_and_m1_m3_mapping(self):
        config = runner.load_config()
        self.assertEqual(config["lambda_values"], [0.0, 0.25, 0.5, 1.0, 2.0])
        self.assertEqual(
            runner.select_lambda_candidate(
                [
                    {"mean_inner_loss": 0.1, "std_inner_loss": 0.2, "evaluation": 1},
                    {"mean_inner_loss": 0.15, "std_inner_loss": 0.01, "evaluation": 2},
                ],
                0.0,
            )["evaluation"],
            1,
        )
        self.assertEqual(
            runner.select_lambda_candidate(
                [
                    {"mean_inner_loss": 0.1, "std_inner_loss": 0.2, "evaluation": 1},
                    {"mean_inner_loss": 0.15, "std_inner_loss": 0.01, "evaluation": 2},
                ],
                1.0,
            )["evaluation"],
            2,
        )
        with self.assertRaises(ValueError):
            runner.select_lambda_candidate([], 0.75)

    def test_lambda_zero_is_mean_loss_objective_and_best_updates(self):
        losses_a = [0.0, 0.0, 0.0, 0.0, 0.4]
        losses_b = [0.15] * 5
        m1_a = runner.HELPERS.fitness_from_losses(losses_a, 0.0)
        m1_b = runner.HELPERS.fitness_from_losses(losses_b, 0.0)
        m3_a = runner.HELPERS.fitness_from_losses(losses_a, 1.0)
        m3_b = runner.HELPERS.fitness_from_losses(losses_b, 1.0)
        self.assertAlmostEqual(m1_a["fitness"], m1_a["mean_loss"])
        self.assertAlmostEqual(m1_b["fitness"], m1_b["mean_loss"])
        self.assertLess(m1_a["fitness"], m1_b["fitness"])
        self.assertGreater(m3_a["fitness"], m3_b["fitness"])
        best, evaluation = runner.HELPERS.update_best(
            m1_a["fitness"], 1, float("inf"), 0
        )
        best, evaluation = runner.HELPERS.update_best(
            m1_b["fitness"], 2, best, evaluation
        )
        self.assertEqual(evaluation, 1)
        self.assertEqual(best, m1_a["mean_loss"])

    def test_nonzero_lambda_changes_best_update_criterion(self):
        losses_variable = [0.0, 0.0, 0.0, 0.0, 0.4]
        losses_stable = [0.15] * 5
        variable_mean = runner.HELPERS.fitness_from_losses(
            losses_variable, 0.0
        )["fitness"]
        stable_mean = runner.HELPERS.fitness_from_losses(
            losses_stable, 0.0
        )["fitness"]
        variable_penalized = runner.HELPERS.fitness_from_losses(
            losses_variable, 1.0
        )["fitness"]
        stable_penalized = runner.HELPERS.fitness_from_losses(
            losses_stable, 1.0
        )["fitness"]
        self.assertLess(variable_mean, stable_mean)
        self.assertGreater(variable_penalized, stable_penalized)

    def test_matched_splits_are_built_once_and_initial_positions_checked(self):
        config = runner.load_config()
        X = pd.DataFrame({"x": range(100)})
        y = pd.Series([0, 1] * 50)
        plans = runner.build_nested_splits(X, y, 44, config)
        self.assertEqual(len(plans), 5)
        for fold, train, test, inner in plans:
            self.assertEqual(len(set(train).intersection(test)), 0)
            self.assertEqual(len(inner), 5)
            self.assertEqual(len(inner[0][0]) + len(inner[0][1]), len(train))
        histories = {
            lam: [
                {
                    "iteration": 1,
                    "particle": p,
                    "initial_log10_C": float(p),
                    "initial_log10_gamma": float(-p),
                }
                for p in range(8)
            ]
            for lam in config["lambda_values"]
        }
        runner.assert_matched_initializations(histories)
        histories[0.5][0]["initial_log10_C"] = 99
        with self.assertRaises(AssertionError):
            runner.assert_matched_initializations(histories)

    def test_candidate_budget_is_48(self):
        self.assertEqual(
            runner.load_config()["evaluations_per_search"],
            runner.HELPERS.N_PARTICLES * runner.HELPERS.N_ITERATIONS,
        )
        history = [{"evaluation": value} for value in range(1, 49)]
        runner.assert_candidate_budget(history, 48)
        with self.assertRaises(AssertionError):
            runner.assert_candidate_budget(history[:-1], 48)

    def test_each_run_uses_explicit_svd_seed(self):
        helper = runner.HELPERS
        original_seed, original_svd_seed = helper.SEED, helper.SVD_RANDOM_STATE
        try:
            for seed in (42, 43, 44):
                helper.SEED = seed
                helper.SVD_RANDOM_STATE = seed
                model = helper.create_model(
                    1.0,
                    1.0,
                    ["feature"],
                    [],
                    {
                        "numeric_imputation": "median",
                        "dimensionality_reduction": "svd",
                    },
                )
                estimator = model.named_steps["preprocessing"].named_steps["reduction"]
                self.assertEqual(estimator.random_state, seed)
        finally:
            helper.SEED, helper.SVD_RANDOM_STATE = original_seed, original_svd_seed

    def test_outer_test_is_not_used_for_search_or_threshold_choice(self):
        source = runner.V2_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn('"outer_test_used_for_search": False', source)
        self.assertIn('"outer_test_used_for_selection": False', source)
        scorer = source[source.index("def score_candidate("):source.index("def append_summary_rows(")]
        self.assertIn("cross_val_predict", scorer)
        self.assertIn("X_train", scorer)
        self.assertIn("y_train", scorer)
        self.assertIn("threshold_training_scope", scorer)

    def test_lambda_pairing_is_grouped_and_computed_against_zero(self):
        rows = []
        for lam, metric in ((0.0, 0.7), (0.25, 0.75), (0.5, 0.8), (1.0, 0.6), (2.0, 0.9)):
            rows.append(
                {
                    "seed": 42,
                    "dataset": "diabetes",
                    "preprocessing_config": "baseline_median",
                    "outer_fold": 1,
                    "lambda": lam,
                    **{name: metric for name in runner.METRICS},
                }
            )
        paired = runner._paired_differences(pd.DataFrame(rows))
        self.assertEqual(len(paired), 4)
        deltas = paired.set_index("lambda")["outer_roc_auc_lambda_minus_0"]
        self.assertAlmostEqual(deltas[0.25], 0.05)
        self.assertAlmostEqual(deltas[0.5], 0.1)
        self.assertAlmostEqual(deltas[1.0], -0.1)
        self.assertAlmostEqual(deltas[2.0], 0.2)
        by_seed, across_seeds = runner._summarize_paired(paired)
        self.assertEqual(len(by_seed), 4)
        self.assertEqual(len(across_seeds), 4)

    def test_output_isolation_refuses_existing_seed_and_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "seed_42"
            output.mkdir()
            sentinel = output / "preserve.txt"
            sentinel.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.require_absent(output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
        self.assertEqual(
            set(runner.AGGREGATE_FILES),
            {
                "paired_lambda_fold_differences.csv",
                "paired_lambda_calibrated_fold_differences.csv",
                "paired_lambda_summary_by_seed.csv",
                "paired_lambda_calibrated_summary_by_seed.csv",
                "paired_lambda_summary_across_seeds.csv",
                "paired_lambda_calibrated_summary_across_seeds.csv",
                "findings_summary.md",
            },
        )

    def test_preflight_allows_only_the_expected_findings_placeholder(self):
        config = runner.load_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            findings = root / "findings_summary.md"
            findings.write_text(
                "# Findings\n\n**Status: Not run.**\n", encoding="utf-8"
            )
            with patch.object(runner, "EXPERIMENT_DIR", root):
                runner.preflight_outputs(config["seeds"])
                (root / "seed_43").mkdir()
                with self.assertRaises(FileExistsError):
                    runner.preflight_outputs(config["seeds"])
                (root / "seed_43").rmdir()
                findings.write_text("# Existing findings\n", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    runner.preflight_outputs(config["seeds"])

    def _paired_result_key_frames(self):
        config = {
            "datasets": ["dataset_a"],
            "preprocessing_configs": [{"name": "prep_a"}],
            "outer_folds": 2,
        }
        rows = [
            {
                "seed": 42,
                "dataset": "dataset_a",
                "preprocessing_config": "prep_a",
                "outer_fold": fold,
                "lambda": lambda_value,
            }
            for fold in (1, 2)
            for lambda_value in sorted(validator.LAMBDAS)
        ]
        return config, pd.DataFrame(rows), pd.DataFrame(rows)

    def test_validator_detects_missing_outer_fold_keys(self):
        config, default, calibrated = self._paired_result_key_frames()
        errors = []
        validator._validate_paired_result_keys(
            default[default["outer_fold"] != 2],
            calibrated,
            config,
            42,
            errors,
        )
        self.assertTrue(
            any("default-threshold results are missing 5 expected keys" in e for e in errors),
            errors,
        )

    def test_validator_detects_missing_calibrated_result_key(self):
        config, default, calibrated = self._paired_result_key_frames()
        errors = []
        validator._validate_paired_result_keys(
            default, calibrated.iloc[:-1], config, 42, errors
        )
        self.assertTrue(
            any("threshold-calibrated results are missing 1 expected keys" in e for e in errors),
            errors,
        )
        self.assertTrue(
            any("result keys do not match" in e for e in errors), errors
        )

    def test_validator_detects_duplicate_result_keys(self):
        config, default, calibrated = self._paired_result_key_frames()
        default = pd.concat([default, default.iloc[[0]]], ignore_index=True)
        errors = []
        validator._validate_paired_result_keys(
            default, calibrated, config, 42, errors
        )
        self.assertIn("Duplicate default-threshold result keys.", errors)

    def test_aggregate_validator_recognizes_existing_dependence_warning(self):
        readme = runner.EXPERIMENT_DIR.joinpath("README.md").read_text(
            encoding="utf-8"
        )
        self.assertTrue(validator._documents_dependence_limitations(readme))

    def test_aggregation_revalidates_saved_files_despite_passed_marker(self):
        config = {"seeds": [42]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_dir = root / "seed_42"
            seed_dir.mkdir()
            (root / "findings_summary.md").write_text(
                "# Findings\n\n**Status: Not run.**\n", encoding="utf-8"
            )
            (seed_dir / "experiment_config.json").write_text(
                json.dumps({"seed": 42, "validation_status": "passed"}),
                encoding="utf-8",
            )
            with patch.object(runner, "EXPERIMENT_DIR", root):
                with self.assertRaisesRegex(RuntimeError, "fresh validation"):
                    runner.aggregate_completed_seeds(config)
            self.assertFalse(
                (root / "paired_lambda_fold_differences.csv").exists()
            )
            report = (seed_dir / "validation_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Missing required files", report)

    def test_seed_validation_promotes_only_after_success(self):
        config = runner.load_config()
        good = Path("seed_42")
        with patch.object(
            validator, "validate_seed_directory", return_value=[]
        ) as validate:
            with patch.object(
                validator.runner, "seed_output_path", return_value=good
            ):
                with tempfile.TemporaryDirectory() as temporary:
                    seed_dir = Path(temporary) / "seed_42"
                    seed_dir.mkdir()
                    completion = pd.DataFrame(
                        {"status": ["validation_pending", "validation_pending"]}
                    )
                    completion.to_csv(seed_dir / "completion_log.csv", index=False)
                    actual = {"validation_status": "in_progress"}
                    (seed_dir / "experiment_config.json").write_text(
                        json.dumps(actual), encoding="utf-8"
                    )
                    with patch.object(
                        validator.runner, "seed_output_path", return_value=seed_dir
                    ):
                        self.assertEqual(validator.promote_seed(42, config), [])
                    saved = pd.read_csv(seed_dir / "completion_log.csv")
                    self.assertTrue(saved["status"].eq("completed").all())
                    self.assertEqual(
                        json.loads(
                            (seed_dir / "experiment_config.json").read_text(
                                encoding="utf-8"
                            )
                        )["validation_status"],
                        "passed",
                    )
                    self.assertEqual(validate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
