from typing import Optional, List

import torch

from bigrams.tokenizer import TokenizerCharacter
from bigrams.viz import print_matrix_section


class TransitionMatrix:
    """
    A transition matrix for character sequences that tracks how often characters follow each other.
    Each row i represents a current character, each column j represents the next character,
    and the value at [i,j] represents how many times character j followed character i in the training data.

    Example for words ["cat", "cab"]:
    - Tokenizer vocabulary: {'.' (special): 0, 'a': 1, 'b': 2, 'c': 3, 't': 4}
    - Encoded words: [0,3,1,4,0] and [0,3,1,2,0]
    - Resulting counts matrix:
      [
        [0, 0, 0, 2, 0],  # After '.' (start), we see 'c' twice
        [0, 0, 1, 0, 1],  # After 'a', we see 'b' once and 't' once
        [2, 0, 0, 0, 0],  # After 'b', we see '.' (end) once
        [0, 2, 0, 0, 0],  # After 'c', we see 'a' twice
        [1, 0, 0, 0, 0],  # After 't', we see '.' (end) once
      ]
    """

    def __init__(self, counts: torch.Tensor, tokenizer: TokenizerCharacter) -> None:
        """
        Initialize transition matrix from counts tensor.

        Args:
            counts: Transition counts tensor of shape [vocab_size, vocab_size]
                   where counts[i,j] represents how many times character j
                   followed character i in the training data.
                   Example for ["cat"]:
                   [[0, 0, 0, 1, 0],  # After '.' -> 'c'
                    [0, 0, 0, 0, 1],  # After 'a' -> 't'
                    [0, 0, 0, 0, 0],  # After 'b' (never seen)
                    [0, 1, 0, 0, 0],  # After 'c' -> 'a'
                    [1, 0, 0, 0, 0]]  # After 't' -> '.'
            tokenizer: Fitted TokenizerCharacter instance that defines the mapping
                      between characters and indices

        Raises:
            ValueError: If counts shape doesn't match tokenizer's vocabulary size
        """
        if counts.shape != (tokenizer.vocab_size, tokenizer.vocab_size):
            raise ValueError(
                f"Counts shape {counts.shape} doesn't match tokenizer vocab size "
                f"{tokenizer.vocab_size}"
            )

        self.counts = counts
        self.tokenizer = tokenizer
        self._probabilities: Optional[torch.Tensor] = None

    @staticmethod
    def from_words(words: List[str], tokenizer: TokenizerCharacter) -> 'TransitionMatrix':
        """
        Create transition matrix by counting character transitions in words.

        Example:
            words = ["cat"]
            tokenizer vocabulary = {'.': 0, 'a': 1, 'c': 2, 't': 3}
            encoded = [0,2,1,3,0]  # [START, 'c', 'a', 't', END]
            transitions = [(0,2), (2,1), (1,3), (3,0)]  # pairs of consecutive indices
            resulting counts[0,2] = 1  # START->c
                           counts[2,1] = 1  # c->a
                           counts[1,3] = 1  # a->t
                           counts[3,0] = 1  # t->END

        Args:
            words: List of words to compute transitions from
            tokenizer: Fitted TokenizerCharacter instance

        Returns:
            TransitionMatrix instance with computed transition counts
        """
        counts = torch.zeros(
            (tokenizer.vocab_size, tokenizer.vocab_size),
            dtype=torch.int32
        )

        for word in words:
            indices = tokenizer.encode_word(word)
            for idx1, idx2 in zip(indices, indices[1:]):
                counts[idx1, idx2] += 1

        return TransitionMatrix(counts, tokenizer)

    @property
    def probabilities(self) -> torch.Tensor:
        """
        Convert counts to probabilities by normalizing each row to sum to 1.
        Computed once and cached for subsequent access.

        Example:
            counts = [[0, 0, 2, 0],    probabilities = [[0.0, 0.0, 1.0, 0.0],
                     [0, 0, 1, 1],  →                   [0.0, 0.0, 0.5, 0.5],
                     [1, 1, 0, 0]]                      [0.5, 0.5, 0.0, 0.0]]

        Returns:
            Tensor where each row sums to 1.0, representing probability distributions
            for transitioning from each character to the next character
        """
        if self._probabilities is None:
            probs = self.counts.float()  # Convert to float for division
            row_sums = probs.sum(dim=1, keepdim=True)  # Sum each row
            row_sums[row_sums == 0] = 1  # Prevent division by zero
            self._probabilities = probs / row_sums  # Normalize each row
        return self._probabilities

    def sample_next(self, current_index: int, generator: Optional[torch.Generator] = None) -> int:
        """
        Sample the next character index based on transition probabilities.

        Example:
            current_index = 1  # 'a'
            probabilities[1] = [0.0, 0.0, 0.0, 1.0]  # After 'a', always see 't'
            → returns 3 ('t')

        Args:
            current_index: Index of the current character
            generator: Optional random number generator for reproducibility

        Returns:
            Index of the sampled next character based on transition probabilities
        """
        probs = self.probabilities[current_index]
        return torch.multinomial(probs, 1, generator=generator).item()

    def generate_sequence(self, max_length: int = 50, seed: Optional[int] = None) -> str:
        """
        Generate a new sequence by sampling transitions until reaching end token.

        Example process:
            1. Start with START token (index 0)
            2. Sample next character based on probabilities[0]
            3. Keep sampling next characters until END token or max_length

            For transition probabilities:
            [[0.0, 0.0, 1.0, 0.0],  # START → always 'c'
             [0.0, 0.0, 0.0, 1.0],  # 'a' → always 't'
             [0.0, 1.0, 0.0, 0.0],  # 'c' → always 'a'
             [1.0, 0.0, 0.0, 0.0]]  # 't' → always END

            Might generate: "cat" by sampling path:
            START(0) → 'c'(2) → 'a'(1) → 't'(3) → END(0)

        Args:
            max_length: Maximum length of generated sequence to prevent infinite loops
            seed: Random seed for reproducible generation

        Returns:
            Generated sequence with start/end tokens removed
        """
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        indices: List[int] = []
        current_idx = 0  # Start token

        while len(indices) < max_length:
            next_idx = self.sample_next(current_idx, generator)
            indices.append(next_idx)
            if next_idx == 0:  # End token
                break
            current_idx = next_idx

        return self.tokenizer.decode_indices(indices)


if __name__ == '__main__':
    words = ["cat", "cab", "can", "car", "bat", "bar", "ban"]
    tokenizer = TokenizerCharacter(words)
    matrix = TransitionMatrix.from_words(words, tokenizer)
    print("Transition probabilities:")
    print_matrix_section(matrix, tokenizer)

    print("\nGenerated words:")
    for i in range(5):
        print(matrix.generate_sequence(seed=i))