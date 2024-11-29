from typing import List, Tuple
import torch
from dataclasses import dataclass
from bigrams.tokenizer import TokenizerCharacter


@dataclass
class WordDataset:
    """
    Handles word data and training pair generation with configurable context window size.

    Attributes:
        words: List of training words
        tokenizer: Character-level tokenizer
    """
    words: List[str]
    tokenizer: TokenizerCharacter

    def training_data(self, block_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build dataset with specified context window size.

        Args:
            block_size: Size of the context window (default=1 for bigram model)

        Returns:
            Tuple of (context_tensor, target_tensor)
        """
        X, Y = [], []
        for word in self.words:
            # Initialize context with start tokens
            context = [0] * block_size  # 0 is the special token ('.')

            # Process each character in the word plus the end token
            for ch in word + '.':
                ix = self.tokenizer.encode_char(ch)
                X.append(context.copy())  # Important to copy the context
                Y.append(ix)
                # Update context window: remove oldest and add new token
                context = context[1:] + [ix]

        return torch.tensor(X), torch.tensor(Y)

    def __getitem__(self, idx) -> str:
        return self.words[idx]

    def __len__(self) -> int:
        return len(self.words)


if __name__ == "__main__":
    # Sample data
    words = ['hello', 'world']

    # Create tokenizer and dataset
    tokenizer = TokenizerCharacter(words)
    dataset = WordDataset(words, tokenizer)

    # Get training data with block size = 3
    X, Y = dataset.training_data(block_size=3)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Print first few examples
    for i in range(6):
        context = [tokenizer.decode_char(int(x)) for x in X[i]]
        target = tokenizer.decode_char(int(Y[i]))
        print(f"Context: {context} → Target: {target}")