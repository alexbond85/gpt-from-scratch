import torch


def calculate_min_loss(T: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> float:
    """
    Calculate minimum achievable loss using true transition probabilities.

    Args:
        T: Real transition matrix [vocab_size, vocab_size]
        xs: Input character indices
        ys: Target character indices

    Returns:
        Minimum possible cross-entropy loss
    """
    # Select true probabilities of transitions that occurred in training
    correct_probs = T[xs, ys]

    # Calculate cross-entropy loss without regularization
    min_loss = -torch.log(correct_probs).mean()

    return min_loss.item()