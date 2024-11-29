import torch
import torch.nn.functional as F
from typing import List, Tuple
from bigrams.tokenizer import TokenizerCharacter
from ngrams_mlp.ngram_nn import NGramNN


class CharacterGenerator:
    """
    A generator class that works with trained NGramNN models to:
    1. Predict next characters
    2. Calculate probabilities of specific sequences
    3. Find top-k most likely next characters
    """

    def __init__(self, model: NGramNN, tokenizer: TokenizerCharacter):
        """
        Initialize generator with trained model and tokenizer.

        Args:
            model: Trained NGramNN model
            tokenizer: Character tokenizer used during training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = model.block_size

    def _prepare_context(self, context: str) -> List[int]:
        """
        Prepare context for model by padding or trimming as needed.

        Args:
            context: Input context of any length

        Returns:
            List of block_size indices, padded with start tokens if needed

        Raises:
            ValueError: If context is longer than block_size
        """
        if len(context) > self.block_size:
            raise ValueError(f"Context cannot be longer than {self.block_size} characters")

        # Pad context with start tokens if needed
        padding_needed = self.block_size - len(context)
        if padding_needed > 0:
            context = '.' * padding_needed + context

        # Convert to indices
        return [self.tokenizer.encode_char(ch) for ch in context]

    def predict_next(self, context: str) -> str:
        """Predict single most likely next character given context."""
        indices = self._prepare_context(context)

        logits = self.model.forward(torch.tensor([indices]))
        probs = F.softmax(logits, dim=-1)

        next_idx = torch.argmax(probs).item()
        return self.tokenizer.decode_char(next_idx)

    def sequence_probability(self, context: str, next_char: str) -> float:
        """Calculate probability of specific next character given context."""
        indices = self._prepare_context(context)
        target_idx = self.tokenizer.encode_char(next_char)

        logits = self.model.forward(torch.tensor([indices]))
        probs = F.softmax(logits, dim=-1)

        return probs[0, target_idx].item()

    def top_k_predictions(self, context: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return top k most likely next characters with their probabilities."""
        indices = self._prepare_context(context)

        logits = self.model.forward(torch.tensor([indices]))
        probs = F.softmax(logits, dim=-1)[0]

        top_probs, top_indices = torch.topk(probs, k)

        return [
            (self.tokenizer.decode_char(idx.item()), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

    def sample_word(self, context: str = "", temperature: float = 1.0, max_length: int = 20) -> str:
        """
        Sample a random word continuation given optional context.

        Args:
            context: Starting characters (optional)
            temperature: Controls randomness (higher = more random)
            max_length: Maximum length including context

        Returns:
            Generated word including the context

        Examples:
            generator.sample_word()           # Completely new word
            generator.sample_word("he")       # Word starting with "he"
            generator.sample_word("hea")      # Word starting with "hea"
        """
        if len(context) > max_length:
            raise ValueError("Context length cannot exceed max_length")

        # Initialize with padded context
        current = self._prepare_context(context)
        sequence = list(self.tokenizer.encode_char(ch) for ch in context if ch != '.')

        # Generate remaining characters
        remaining_length = max_length - len(context)

        for _ in range(remaining_length):
            # Get model predictions
            logits = self.model.forward(torch.tensor([current]))
            probs = F.softmax(logits / temperature, dim=-1)

            # Sample next character
            next_idx = torch.multinomial(probs[0], num_samples=1).item()

            # Stop if we generate the end token
            if next_idx == 0:
                break

            sequence.append(next_idx)
            # Update context window
            current = current[1:] + [next_idx]

        return self.tokenizer.decode_indices(sequence)

    def sample_words(self, n: int = 5, context: str = "", temperature: float = 1.0, max_length: int = 20) -> List[str]:
        """
        Generate multiple random words with the same context.

        Args:
            n: Number of words to generate
            context: Starting characters (optional)
            temperature: Sampling temperature
            max_length: Maximum word length

        Returns:
            List of generated words
        """
        return [self.sample_word(context, temperature, max_length) for _ in range(n)]
