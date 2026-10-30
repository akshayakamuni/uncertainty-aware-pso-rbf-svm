
import numpy as np


# Predefine these values before the final evaluation.
LAMBDA_VALUES = [0.25, 0.5, 1.0, 2.0]

# Primary value for the initial experiment.
DEFAULT_LAMBDA = 1.0


def calculate_candidate_statistics(fold_losses):
    """
    Calculate mean loss and fold-to-fold variability
    for every candidate.

    Parameters
    ----------
    fold_losses : array-like
        Shape: (number of candidates, number of folds).
        Lower loss is better.

    Returns
    -------
    mean_losses : numpy array
    std_losses : numpy array
    """

    losses = np.asarray(fold_losses, dtype=float)

    if losses.ndim != 2:
        raise ValueError(
            "fold_losses must be a 2D array: candidates x folds."
        )

    if losses.shape[0] == 0 or losses.shape[1] < 2:
        raise ValueError(
            "Provide at least one candidate and two folds."
        )

    if not np.isfinite(losses).all():
        raise ValueError("fold_losses contains invalid values.")

    mean_losses = losses.mean(axis=1)

    # Sample standard deviation across folds.
    std_losses = losses.std(axis=1, ddof=1)

    return mean_losses, std_losses


def select_standard_candidate(fold_losses):
    """Choose the candidate with the lowest mean loss."""

    mean_losses, std_losses = calculate_candidate_statistics(
        fold_losses
    )

    best_index = int(np.argmin(mean_losses))

    return {
        "best_index": best_index,
        "mean_loss": float(mean_losses[best_index]),
        "std_loss": float(std_losses[best_index]),
    }


def select_uncertainty_aware_candidate(
    fold_losses,
    lambda_value=DEFAULT_LAMBDA,
):
    """
    Choose the candidate minimizing:

        score = mean loss + lambda * standard deviation
    """

    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative.")

    mean_losses, std_losses = calculate_candidate_statistics(
        fold_losses
    )

    scores = mean_losses + lambda_value * std_losses
    best_index = int(np.argmin(scores))

    return {
        "best_index": best_index,
        "mean_loss": float(mean_losses[best_index]),
        "std_loss": float(std_losses[best_index]),
        "selection_score": float(scores[best_index]),
        "lambda_value": float(lambda_value),
    }


if __name__ == "__main__":
    # Illustrative example only, not real dataset results.
    example_losses = [
        [0.16, 0.18, 0.20, 0.17, 0.19],  # Candidate A
        [0.10, 0.22, 0.14, 0.19, 0.19],  # Candidate B
        [0.19, 0.20, 0.21, 0.20, 0.20],  # Candidate C
    ]

    mean_losses, std_losses = calculate_candidate_statistics(
        example_losses
    )

    print("Illustrative candidate results")
    print("-" * 50)

    for i, (mean_loss, std_loss) in enumerate(
        zip(mean_losses, std_losses)
    ):
        print(
            f"Candidate {i}: "
            f"mean loss = {mean_loss:.4f}, "
            f"standard deviation = {std_loss:.4f}"
        )

    standard = select_standard_candidate(example_losses)
    print("\nStandard selection:", standard)

    for lambda_value in LAMBDA_VALUES:
        result = select_uncertainty_aware_candidate(
            example_losses,
            lambda_value=lambda_value,
        )
        print(
            f"Uncertainty-aware selection "
            f"(lambda={lambda_value}):",
            result,
        )
