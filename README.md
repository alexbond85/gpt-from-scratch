# Neural Character Generation Models

This project implements character-level language models using neural networks, demonstrating progression from simple bigram models to more advanced architectures.

## Project Structure

- `bigrams/`: Simple character bigram models
  - Statistical transition matrix
  - Neural transition matrix
  - See `bigrams/README.md` for details

- `ngrams_mlp/`: Advanced n-gram neural network
  - Configurable context window
  - Character embeddings
  - See `ngrams_mlp/README.md` for details

## Quick Start

```python
from bigrams.tokenizer import TokenizerCharacter
from bigrams.trmatrix import TransitionMatrix
from bigrams.trmatrix_nn import NeuralTransitionMatrix
from bigrams.word_dataset import WordDatasetBigram
from ngrams_mlp.chargen import CharacterGenerator
from ngrams_mlp.word_dataset import WordDataset
from ngrams_mlp.ngram_nn import NGramNN

# Prepare data
words = ["cat", "car", "cab"]
tokenizer = TokenizerCharacter(words)
dataset_bigram = WordDatasetBigram(words, tokenizer)
dataset_ngram = WordDataset(words, tokenizer)

# 1. Statistical bigram model
stats_model = TransitionMatrix.from_words(words, tokenizer)
print("Statistical bigram:", stats_model.generate_sequence())  # e.g. "cat"

# 2. Neural bigram model
neural_model = NeuralTransitionMatrix(tokenizer)
neural_model.train_bigram_model(dataset_bigram)
print("Neural bigram:", neural_model.generate_sequence())  # e.g. "car"

# 3. Advanced n-gram model with larger context
ngram_model = NGramNN(
    tokenizer=tokenizer,
    block_size=3,          # Use 3 characters of context
    embedding_dim=2        # Represent each character as 2D vector
)
ngram_model.train(dataset_ngram, num_epochs=100)
bgram_generator = CharacterGenerator(ngram_model, tokenizer)
print("3-gram neural:", bgram_generator.sample_words())  
```
