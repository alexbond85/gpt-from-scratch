import torch
import torch.nn.functional as F
from typing import Optional, List
from bigrams.tokenizer import TokenizerCharacter
from ngrams_mlp.word_dataset import WordDataset


class NGramNN:
    """
        Neural network for character-level language modeling using n-gram context.

        Architecture:
        1. Character Embedding Layer: Converts characters to dense vectors
        2. Hidden Layer: Learns patterns with tanh activation
        3. Output Layer: Produces next-character probabilities

        Example for word "cat":
        - Input contexts: ['..','.c','ca']
        - Targets:        ['c','a','t']
        Each context is embedded, concatenated, and processed to predict next char.
        """
    def __init__(self,
                 tokenizer: TokenizerCharacter,
                 block_size: int = 3,
                 embedding_dim: int = 2,
                 hidden_dim: int = 100,
                 generator: Optional[torch.Generator] = None):
        """
        Initialize the model parameters.

        Args:
            tokenizer: Character tokenizer defining the vocabulary
            block_size: Number of context characters to use (n-1 in n-gram)
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden layer
            generator: Random number generator for reproducibility
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.embedding_dim = embedding_dim

        # Initialize parameters with specified generator
        g = generator or torch.Generator().manual_seed(2147483647)

        # Create parameters that require gradients
        # Character embedding matrix: maps each character to embedding_dim vector
        # Example: 'a' (index 1) -> [0.5, -0.3] for embedding_dim=2
        self.C = torch.nn.Parameter(torch.randn((tokenizer.vocab_size, embedding_dim), generator=g))

        # Hidden layer weights: process concatenated embeddings
        # Input dim: block_size * embedding_dim (concatenated context vectors)
        # Output dim: hidden_dim (learned internal representations)
        self.W1 = torch.nn.Parameter(torch.randn((block_size * embedding_dim, hidden_dim), generator=g))
        self.b1 = torch.nn.Parameter(torch.randn(hidden_dim, generator=g))
        # Output layer: map hidden states to character probabilities
        # Input dim: hidden_dim
        # Output dim: vocab_size (score for each possible next character)
        self.W2 = torch.nn.Parameter(torch.randn((hidden_dim, tokenizer.vocab_size), generator=g))
        self.b2 = torch.nn.Parameter(torch.randn(tokenizer.vocab_size, generator=g))

        # Collect all parameters for optimizer
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Process flow:
        1. Embed: Convert each character in context to vector
        2. Concat: Combine context character embeddings
        3. Hidden: Apply non-linear transformation
        4. Output: Compute scores for each possible next character

        Example dimensions for batch_size=32, block_size=3, embedding_dim=2:
        X: [32, 3] -> emb: [32, 3, 2] -> concat: [32, 6] ->
        h: [32, 100] -> logits: [32, vocab_size]
        """
        # 0. Start with character indices
        # X = torch.tensor([[0, 3, 1], [1, 2, 3], [2, 1, 4], [3, 3, 2]])
        # [batch_size=4, block_size=3]
        #
        # 1. Embedding lookup: convert character indices to vectors
        # emb = [
        #     # Batch item 1
        #     [
        #         [0.1, 0.2],  # First character embedding
        #         [0.3, 0.4],  # Second character embedding
        #         [0.5, 0.6]   # Third character embedding
        #     ],
        #     # Batch item 2
        #     [
        #         [0.7, 0.8],
        #         [0.9, 1.0],
        #         [1.1, 1.2]
        #     ],
        #     # ... 2 more batch items
        # ]
        # Shape: [batch_size, block_size, embedding_dim]
        emb = self.C[X]

        # 2. Concatenate embeddings of context characters into single vector
        # Shape: [batch_size, block_size * embedding_dim]
        # Transforms each sequence from:
        #   [[e1.x, e1.y], [e2.x, e2.y], [e3.x, e3.y]]
        # To:
        #   [e1.x, e1.y, e2.x, e2.y, e3.x, e3.y]
        # concat = [
        #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Batch item 1 flattened
        #     [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],  # Batch item 2 flattened
        #     ... 2 more batch items
        # ]
        # Shape: [batch_size, block_size * embedding_dim]
        concat = emb.view(-1, self.block_size * self.embedding_dim)

        # Hidden layer with tanh activation
        h = torch.tanh(concat @ self.W1 + self.b1)

        # Output layer (logits)
        logits = h @ self.W2 + self.b2

        return logits

    def train(self, dataset: WordDataset,
              learning_rate: float = 0.1,
              num_epochs: int = 200) -> List[float]:
        """Train the model using the provided dataset."""
        optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        losses = []

        for epoch in range(num_epochs):
            # Get training data using dataset's method
            X, Y = dataset.training_data(self.block_size)

            # Forward pass
            logits = self.forward(X)
            loss = F.cross_entropy(logits, Y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track progress
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            losses.append(loss.item())

        return losses


if __name__ == "__main__":
    # Sample data
    words = ['hello', 'world', 'neural', 'network', 'test']

    # Create tokenizer and dataset
    tokenizer = TokenizerCharacter(words)
    dataset = WordDataset(words, tokenizer)

    # Create model with specified parameters
    model = NGramNN(
        tokenizer=tokenizer,
        block_size=3,
        embedding_dim=2,
        generator=torch.Generator().manual_seed(2147483647)
    )

    # Train the model
    print("Training model...")
    losses = model.train(dataset, learning_rate=0.1, num_epochs=10)

    # Generate some words
    print("\nGenerated words:")
    for _ in range(5):
        word = model.generate()
        print(word)