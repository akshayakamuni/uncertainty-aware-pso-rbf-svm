"""Lightweight tests for the multi-seed runner's safeguards and pairing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from unittest.mock import patch

import run_multiseed
import validate_multiseed


class MultiSeedRunnerTests(unittest.TestCase):
    def test_only_m1_and_m3_are_selected_with_lambda_one(self):
        m1_pool = [
            {"mean_inner_loss": 0.10, "std_inner_loss": 0.08, "evaluation": 1},
            {"mean_inner_loss": 0.12, "std_inner_loss": 0.01, "evaluation": 2},
        ]
        m3_pool = [
            {"mean_inner_loss": 0.10, "std_inner_loss": 0.08, "evaluation": 1},
            {"mean_inner_loss": 0.12, "std_inner_loss": 0.01, "evaluation": 2},
        ]
        m1, m3 = run_multiseed.select_method_candidates(m1_pool, m3_pool)
        self.assertEqual(m1["evaluation"], 1)
        self.assertEqual(m3["evaluation"], 2)
        self.assertEqual(run_multiseed.load_config()["methods"], [
            "M1 Standard PSO",
            "M3 UA-PSO",
        ])
        with self.assertRaises(ValueError):
            run_multiseed.select_method_candidates(m1_pool, m3_pool, 0.5)

    def test_svd_pipeline_uses_each_run_seed(self):
        helper = run_multiseed.HELPERS
        original_seed = helper.SEED
        original_svd_seed = helper.SVD_RANDOM_STATE
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
                reduction = model.named_steps["preprocessing"].named_steps["reduction"]
                self.assertEqual(reduction.random_state, seed)
        finally:
            helper.SEED = original_seed
            helper.SVD_RANDOM_STATE = original_svd_seed

    def test_matched_initial_positions_are_checked_explicitly(self):
        m1_history = [
            {
                "iteration": 1,
                "particle": index,
                "initial_log10_C": float(index),
                "initial_log10_gamma": float(-index),
            }
            for index in range(8)
        ]
        m3_history = [dict(row) for row in m1_history]
        run_multiseed.assert_matched_initializations(m1_history, m3_history)
        m3_history[-1]["initial_log10_C"] += 0.01
        with self.assertRaises(AssertionError):
            run_multiseed.assert_matched_initializations(m1_history, m3_history)

    def test_matched_searches_receive_the_same_training_and_inner_splits(self):
        history = [
            {
                "iteration": (index - 1) // 8 + 1,
                "particle": (index - 1) % 8,
                "evaluation": index,
                "initial_log10_C": float((index - 1) % 8),
                "initial_log10_gamma": float(-((index - 1) % 8)),
            }
            for index in range(1, 49)
        ]
        inner_splits = [(object(), object())]
        X_train, y_train = object(), object()
        with patch.object(
            run_multiseed.HELPERS,
            "run_pso_search",
            side_effect=[
                (history, 1.0),
                ([dict(row) for row in history], 1.0),
            ],
        ) as search:
            run_multiseed.run_matched_pso_searches(
                X_train,
                y_train,
                ["feature"],
                [],
                {"name": "baseline_median"},
                "diabetes",
                1,
                43,
                [],
                inner_splits,
                48,
            )

        calls = search.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0].args[0], X_train)
        self.assertIs(calls[1].args[0], X_train)
        self.assertIs(calls[0].args[1], y_train)
        self.assertIs(calls[1].args[1], y_train)
        self.assertIs(calls[0].args[-1], inner_splits)
        self.assertIs(calls[1].args[-1], inner_splits)
        self.assertEqual([call.args[7] for call in calls], [0.0, 1.0])

    def test_candidate_budget_and_unique_evaluation_ids(self):
        history = [{"evaluation": value} for value in range(1, 49)]
        run_multiseed.assert_candidate_budget(history, 48)
        with self.assertRaises(AssertionError):
            run_multiseed.assert_candidate_budget(history[:-1], 48)
        history[-1]["evaluation"] = 47
        with self.assertRaises(AssertionError):
            run_multiseed.assert_candidate_budget(history, 48)

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "seed_42"
            existing.mkdir()
            sentinel = existing / "preserve.txt"
            sentinel.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_multiseed.require_output_path_absent(existing)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_result_pairing_uses_seed_dataset_config_and_fold(self):
        rows = []
        for seed in (42, 43):
            for method, auc in (
                ("M1 Standard PSO", 0.70),
                ("M3 UA-PSO", 0.75),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "dataset": "diabetes",
                        "preprocessing_config": "baseline_median",
                        "outer_fold": 1,
                        "method": method,
                        **{metric: auc for metric in run_multiseed.METRICS},
                    }
                )
        paired = run_multiseed.build_paired_differences(pd.DataFrame(rows))
        self.assertEqual(len(paired), 2)
        self.assertEqual(set(paired["seed"]), {42, 43})
        self.assertTrue(
            (paired["outer_roc_auc_M3_minus_M1"].sub(0.05).abs() < 1e-12).all()
        )

    def test_candidate_selection_validation_tolerates_csv_float_roundtrip_ties(self):
        results = pd.DataFrame(
            [
                {
                    "seed": 42,
                    "dataset": "diabetes",
                    "preprocessing_config": "baseline_median",
                    "outer_fold": 1,
                    "method": "M1 Standard PSO",
                    "selected_evaluation": 33,
                    "C": 1.0,
                    "gamma": 0.1,
                }
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "seed": 42,
                    "dataset": "diabetes",
                    "preprocessing_config": "baseline_median",
                    "outer_fold": 1,
                    "method": "M1 Standard PSO",
                    "evaluation": evaluation,
                    "C": 1.0,
                    "gamma": 0.1,
                    "mean_inner_loss": (
                        0.0041279669762642 if evaluation in {15, 33} else 0.1
                    ),
                    "std_inner_loss": 0.0046 if evaluation == 15 else 0.0049,
                }
                for evaluation in range(1, 49)
            ]
        )
        errors = []
        validate_multiseed._validate_candidate_selection(
            results, candidates, errors
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
