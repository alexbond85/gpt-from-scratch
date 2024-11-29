# NGram MLP Neural Network for Character Generation

## Motivation

Traditional n-gram models for character-level text generation face significant challenges:

1. **Limited Context**: Bi-gram models only capture single-character dependencies (P(character|previous_character)), missing longer patterns in text.

2. **Exponential Complexity**: Extending to higher-order n-grams (3-gram, 4-gram) becomes exponentially complex:
   - 2-gram with 26 letters: 26² = 676 parameters
   - 3-gram with 26 letters: 26³ = 17,576 parameters
   - 4-gram with 26 letters: 26⁴ = 456,976 parameters

3. **Sparsity**: As n increases, many possible character combinations never appear in training data, leading to poor generalization.

This project implements a neural network approach that solves these problems by:
- Using learned embeddings to represent characters in a dense, low-dimensional space
- Supporting configurable context window sizes without exponential parameter growth
- Learning smooth probability distributions that generalize better to unseen sequences

## Architecture

The model consists of three main components:

1. **Character Embedding Layer**: Maps each character to a dense vector (e.g., dimension=2)
2. **Hidden Layer**: Processes concatenated context embeddings with non-linear activation
3. **Output Layer**: Produces probabilities for next character prediction

```python
# Example: Creating a 3-gram model with 2D embeddings
model = NGramNN(
    tokenizer=tokenizer,
    block_size=3,          # Use 3 characters of context
    embedding_dim=2,       # Represent each character as 2D vector
    hidden_dim=100         # Internal representation size
)
```

## Features

The `CharacterGenerator` class provides various ways to interact with the trained model:

1. **Next Character Prediction**:
```python
generator = CharacterGenerator(model, tokenizer)
next_char = generator.predict_next("he")  # Might predict 'l'
```

2. **Sequence Probability**:
```python
# How likely is 'l' after 'he'?
prob = generator.sequence_probability("he", "l")  # e.g., 0.75
```

3. **Top-K Predictions**:
```python
# Get most likely continuations
predictions = generator.top_k_predictions("he", k=3)
# Might return: [('l', 0.75), ('r', 0.15), ('n', 0.10)]
```

4. **Word Generation**:
```python
# Generate complete words
words = generator.sample_words(n=5, context="he", temperature=1.0)
# Might generate: ['hello', 'help', 'heart', 'hedge', 'health']
```

## Advantages

1. **Configurable Context**: Set desired context window size at initialization without exponential parameter growth
2. **Efficient Learning**: Character embeddings capture similarities and patterns
3. **Better Generalization**: Smooth probability distributions handle unseen sequences
4. **Temperature Control**: Adjust randomness in generation from conservative to creative
5. **Multiple Interfaces**: Various ways to interact with model predictions

## Usage

1. Prepare your data and create a tokenizer:
```python
words = ['hello', 'world', 'neural', 'network']
tokenizer = TokenizerCharacter(words)
```

2. Create and train the model:
```python
model = NGramNN(tokenizer=tokenizer, block_size=3)
dataset = WordDataset(words, tokenizer)
model.train(dataset, num_epochs=100)
```

3. Initialize generator and use its methods:
```python
generator = CharacterGenerator(model, tokenizer)

# Generate new words
print(generator.sample_words(5))

# Check specific probabilities
print(generator.sequence_probability("ne", "u"))

# Get top predictions
print(generator.top_k_predictions("ne", k=3))
```

## Customization

The model can be customized through several parameters:
- `block_size`: Length of context window (n-1 in n-gram terms), set at initialization
- `embedding_dim`: Size of character embedding vectors
- `hidden_dim`: Size of hidden layer
- `temperature`: Controls randomness in generation (higher = more creative)

This flexibility allows experimentation with different model capacities and generation styles while maintaining computational efficiency.