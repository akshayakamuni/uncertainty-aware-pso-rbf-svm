"""Focused tests for uncertainty-aware PSO fitness and best updates."""

import unittest

import numpy as np

from run_experiment import (
    SEED,
    create_model,
    fitness_from_losses,
    update_best,
)


class PenalizedFitnessTests(unittest.TestCase):
    def test_svd_pipeline_uses_the_configured_seed(self):
        model = create_model(
            C=1.0,
            gamma=1.0,
            numeric_features=["feature"],
            categorical_features=[],
            config={
                "numeric_imputation": "median",
                "dimensionality_reduction": "svd",
            },
        )

        reduction = model.named_steps["preprocessing"].named_steps["reduction"]
        self.assertEqual(reduction.random_state, SEED)

    def test_sample_standard_deviation_drives_personal_and_global_bests(self):
        mean_better = np.array([0.00, 0.00, 0.00, 0.00, 0.40])
        stable = np.array([0.15, 0.15, 0.15, 0.15, 0.15])
        mean_fitness = fitness_from_losses(mean_better, 0.0)
        penalized_fitness = fitness_from_losses(mean_better, 1.0)
        stable_fitness = fitness_from_losses(stable, 1.0)

        self.assertLess(mean_fitness["fitness"], stable_fitness["mean_loss"])
        self.assertGreater(penalized_fitness["fitness"], stable_fitness["fitness"])

        personal_fitness, personal_eval = update_best(
            penalized_fitness["fitness"], 2, 0.20, 1
        )
        global_fitness, global_eval = update_best(
            penalized_fitness["fitness"], 2, 0.20, 1
        )
        personal_fitness, personal_eval = update_best(
            stable_fitness["fitness"], 3, personal_fitness, personal_eval
        )
        global_fitness, global_eval = update_best(
            stable_fitness["fitness"], 3, global_fitness, global_eval
        )

        self.assertEqual(personal_eval, 3)
        self.assertEqual(global_eval, 3)
        self.assertEqual(personal_fitness, stable_fitness["fitness"])
        self.assertEqual(global_fitness, stable_fitness["fitness"])

    def test_invalid_fold_losses_raise(self):
        with self.assertRaises(ValueError):
            fitness_from_losses([0.1, 0.2, np.nan, 0.2, 0.1], 1.0)


if __name__ == "__main__":
    unittest.main()
