from typing import List, Optional
import torch
import torch.nn.functional as F

from bigrams.loss import calculate_min_loss
from bigrams.tokenizer import TokenizerCharacter
from bigrams.trmatrix import TransitionMatrix
from bigrams.viz import print_matrix_section
from bigrams.word_dataset import WordDataset


class NeuralTransitionMatrix:
    """
    A neural network-based transition matrix that learns character sequence probabilities
    through gradient descent, in contrast to the counting-based statistical approach.

    Core advantages over statistical approach:
    1. Can learn more nuanced patterns by optimizing probability distributions
    2. Uses regularization to prevent overfitting on rare sequences
    3. Can generalize better to unseen combinations

    The neural network architecture:
    - Single weight matrix W of shape [vocab_size, vocab_size]
    - Each row i represents the learned transition probabilities from character i
    - After softmax, W[i,j] represents P(next=j|current=i)

    Example of learned patterns for ["cat", "car"]:
    1. After 'c', network learns high probabilities for both 'a' and 'r'
    2. After 'a', probabilities split between 't' and 'r'
    3. Network can potentially generalize to generate "car" even if only "cat" seen
    """

    def __init__(self, tokenizer: TokenizerCharacter) -> None:
        """
        Initialize transition matrix with random weights.

        Args:
            tokenizer: Character tokenizer defining the vocabulary
                      Vocabulary must include special START/END token (usually '.')

        The weights are initialized with small random values (*0.1) to:
        1. Break symmetry in the network
        2. Start close to uniform probabilities after softmax
        3. Allow gradual learning of transition patterns
        """
        self.tokenizer = tokenizer
        self.weights = torch.randn((tokenizer.vocab_size, tokenizer.vocab_size)) * 0.1
        self._probabilities: Optional[torch.Tensor] = None

    @property
    def probabilities(self) -> torch.Tensor:
        """
        Convert raw weights to transition probabilities using softmax.

        The softmax operation:
        1. Exponentiates weights: exp(w_ij)
        2. Normalizes rows: p_ij = exp(w_ij) / sum(exp(w_i))

        Returns cached probabilities if already computed.

        Example conversion:
        weights = [[1.0, 2.0],     probabilities = [[0.27, 0.73],
                  [0.5, 0.5]]  →                   [0.5,  0.5 ]]
        """
        if self._probabilities is None:
            self._probabilities = F.softmax(self.weights, dim=1)
        return self._probabilities

    def train_bigram_model(self, words: WordDataset, learning_rate: float = 0.1, num_epochs: int = 200) -> None:
        """
        Train the transition matrix using neural network optimization.

        Training process:
        1. Create training pairs from words:
           "cat" → ['.','c'], ['c','a'], ['a','t'], ['t','.']
        2. Convert characters to indices using tokenizer
        3. For each epoch:
           - Convert current chars to one-hot vectors
           - Multiply by weights to get next-char scores
           - Apply softmax to get probabilities
           - Compute loss as negative log probability of correct next chars
           - Add L2 regularization (0.001 * weights²) to prevent overfitting
           - Update weights using Adam optimizer

        Args:
            words: WordDataset
            learning_rate: Controls size of weight updates
                         Higher → faster learning but may overshoot
                         Lower → slower but more stable learning
            num_epochs: Number of times to process all training pairs

        Example for word "cat":
        1. Encoded: [0,3,1,4,0] (START,c,a,t,END)
        2. Training pairs: (0,3), (3,1), (1,4), (4,0)
        3. Goal: maximize P(c|START), P(a|c), P(t|a), P(END|t)
        """
        # Create training pairs from words
        xs, ys = words.training_data()

        # Set up trainable weights and optimizer
        W = torch.nn.Parameter(self.weights)
        optimizer = torch.optim.Adam([W], lr=learning_rate)
        correct_indices = torch.arange(len(ys))
        for epoch in range(num_epochs):
            # Forward pass: convert indices → probabilities
            xenc = F.one_hot(xs, num_classes=self.tokenizer.vocab_size).float()
            logits = xenc @ W
            probs = F.softmax(logits, dim=1)

            # Compute loss: -log(probability of correct next char)
            loss = -torch.log(probs[correct_indices, ys]).mean() # + 0.001 * (W ** 2).mean()

            # Backward pass: compute gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        self.weights = W.detach()
        self._probabilities = None  # Reset cached probabilities

    def sample_next(self, current_index: int, generator: Optional[torch.Generator] = None) -> int:
        """
        Sample next character index based on learned probabilities.

        Process:
        1. Get probability distribution for current character
        2. Sample next index using multinomial distribution

        Example:
        current_index = 1 ('a')
        probabilities[1] = [0.1, 0.0, 0.0, 0.8, 0.1]
                           END   a    b    t    c
        → Likely samples 't' (index 3) but occasionally others

        Args:
            current_index: Index of current character in sequence
            generator: Optional random generator for reproducibility
        """
        probs = self.probabilities[current_index]
        return torch.multinomial(probs, 1, generator=generator).item()

    def generate_sequence(self, max_length: int = 50, seed: Optional[int] = None) -> str:
        """
        Generate new sequence by sampling from learned transitions.

        Generation process:
        1. Start with START token (index 0)
        2. Sample next character using learned probabilities
        3. Repeat until END token or max_length reached
        4. Convert indices back to characters

        Example generation trace:
        START(0) → sample → 'c'(3) → sample → 'a'(1) →
        sample → 't'(4) → sample → END(0) → "cat"

        Args:
            max_length: Maximum sequence length (prevent infinite loops)
            seed: Random seed for reproducible generation
        """
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        indices = []
        current_idx = 0  # Start with START token

        while len(indices) < max_length:
            next_idx = self.sample_next(current_idx, generator)
            indices.append(next_idx)
            if next_idx == 0:  # End token
                break
            current_idx = next_idx

        return self.tokenizer.decode_indices(indices)


