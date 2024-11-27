import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


def read_words(filename: str) -> List[str]:
    """Load words from a file, one per line."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f]


def create_datasets(words: List[str], stoi: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training datasets from words by converting characters to their numerical indices.
    This process creates input-target pairs for training the model.

    Example Workflow:
    For word "cat":
    1. Add markers: "." + "cat" + "." -> ['.', 'c', 'a', 't', '.']
    2. Create pairs: ('.','c'), ('c','a'), ('a','t'), ('t','.')
    3. Convert to indices: (0,3), (3,1), (1,20), (20,0)
    4. Add to xs and ys lists respectively

    Note: The '.' character serves as both start and end marker
    """
    xs, ys = [], []
    for w in words:
        # Add start (.) and end (.) markers to create context
        chars = ['.'] + list(w) + ['.']
        # Create pairs of consecutive characters
        for ch1, ch2 in zip(chars, chars[1:]):
            xs.append(stoi[ch1])  # Current character index
            ys.append(stoi[ch2])  # Next character index
    return torch.tensor(xs), torch.tensor(ys)


def train_bigram_model(xs: torch.Tensor, ys: torch.Tensor, vocab_size: int,
                       learning_rate: float = 0.1, num_epochs: int = 200) -> torch.Tensor:
    """Training Process Details:
    1. Initialize weights as nn.Parameter for proper gradient tracking
    2. Convert inputs to one-hot vectors
    3. Compute logits and apply log_softmax
    4. Calculate loss and update weights"""

    W = torch.nn.Parameter(torch.randn((vocab_size, vocab_size)) * 0.1)
    optimizer = torch.optim.Adam([W], lr=learning_rate)

    for epoch in range(num_epochs):
        xenc = F.one_hot(xs, num_classes=vocab_size).float()
        logits = xenc @ W
        probs = F.softmax(logits, dim=1)  # Shape: [N, vocab_size]

        # Get indices for selecting correct probabilities
        correct_indices = torch.arange(len(ys))  # Shape: [N], contains [0,1,2,...,N-1]
        correct_next_chars = ys  # Shape: [N], contains target character indices

        # Select probability of correct next char for each position
        correct_probs = probs[correct_indices, correct_next_chars]  # Shape: [N]
        loss = -torch.log(correct_probs).mean() + 0.001 * (W ** 2).mean()
        # Zero out gradients from previous backward pass to avoid accumulation
        optimizer.zero_grad()

        # Compute gradients of loss with respect to weights W
        loss.backward()

        # Update weights using computed gradients: W = W - learning_rate * gradients
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    W = F.softmax(W, dim=1)
    return W


def compute_transition_matrix(words: List[str], stoi: Dict[str, int], vocab_size: int) -> torch.Tensor:
    """
    Compute transition matrix from training data.

    The transition matrix M[i,j] represents P(char_j | char_i), estimated as:
    count(char_i followed by char_j) / count(char_i)

    Adds smoothing to avoid zero probabilities.
    """
    trans_matrix = torch.zeros((vocab_size, vocab_size))

    # Count transitions in data
    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1, ix2 = stoi[ch1], stoi[ch2]
            trans_matrix[ix1, ix2] += 1

    # Add small constant for smoothing
    trans_matrix += 1e-3
    row_sums = trans_matrix.sum(dim=1, keepdim=True)
    return trans_matrix / row_sums


def generate_words(model: torch.Tensor, stoi: Dict[str, int], itos: Dict[int, str],
                   num_words: int = 5, method: str = 'neural', temperature: float = 1.0) -> List[str]:
    """
    Generate words using either neural model or transition matrix.

    Args:
        temperature: Controls randomness in sampling
                    - T > 1: More random, diverse outputs
                    - T < 1: More focused, conservative outputs
                    - T = 1: Standard sampling
    """
    vocab_size = len(stoi)
    g = torch.Generator().manual_seed(2147483647)
    words = []

    for _ in range(num_words):
        out = []
        ix = 0  # Start with '.' token
        while True:
            # Get probabilities from model
            if method == 'neural':
                xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
                logits = xenc @ model
                p = F.softmax(logits / temperature, dim=1)
            else:
                p = model[ix]
                p = p.pow(1 / temperature)
                p = p / p.sum()

            # Sample next character
            ix = torch.multinomial(p, num_samples=1, generator=g).item()
            if ix == 0:  # End token
                break
            out.append(itos[ix])
        words.append(''.join(out))
    return words


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



def main():
    # Load data and initialize
    words = read_words('../names.txt')
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(stoi)

    xs, ys = create_datasets(words, stoi)
    print(f"Created dataset with {len(xs)} examples")

    print("\nComputing transition matrix...")
    T = compute_transition_matrix(words, stoi, vocab_size)
    min_loss = calculate_min_loss(T, xs, ys)
    print(f"Minimum achievable loss: {min_loss:.4f}")

    print("\nTraining transition matrix via backprop...")
    W = train_bigram_model(xs, ys, vocab_size)
    trained_loss = calculate_min_loss(W, xs, ys)
    print(f"Trained model loss: {trained_loss:.4f}")

    # Generate with different temperatures to show the effect
    for temp in [0.8, 1.0, 1.2]:
        print(f"\nTemperature {temp}:")
        print("Neural model:", ', '.join(generate_words(W, stoi, itos, temperature=temp)))
        print("Transition matrix:", ', '.join(generate_words(T, stoi, itos, method='transition', temperature=temp)))


if __name__ == '__main__':
    main()