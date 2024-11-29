# Bigram Character Model Implementation

A Python implementation of character-level bigram models using both statistical and neural network approaches for generating text sequences.

## Overview

This project implements two approaches to learning character transitions for text generation:

1. Statistical counting-based transition matrix
2. Neural network-based learned transition probabilities

Both models operate at the character level, learning the probability of each character following another character in the training data.

## Components

### TokenizerCharacter

Handles conversion between characters and indices:
```python
tokenizer = TokenizerCharacter(["cat", "dog"])
indices = tokenizer.encode_word("cat")  # [0, 3, 1, 20, 0]
word = tokenizer.decode_indices(indices)  # "cat"
```

- Uses special token '.' for both START and END markers
- Builds vocabulary from unique characters in training data
- Maps each character to a unique integer index

### TransitionMatrix

Statistical approach using counted transitions:
```python
matrix = TransitionMatrix.from_words(["cat", "car"], tokenizer)
generated = matrix.generate_sequence()  # e.g. "cat"
```

- Counts how often each character follows another in training data
- Normalizes counts into transition probabilities
- Generates new words by sampling from learned probabilities

### NeuralTransitionMatrix 

Neural network approach using gradient descent:
```python
matrix = NeuralTransitionMatrix(tokenizer)
matrix.train_bigram_model(words_dataset)
generated = matrix.generate_sequence()  # e.g. "car"
```

- Learns transition probabilities through optimization
- Uses softmax to convert weights to probabilities

### WordDataset

Handles training data preparation:
```python
dataset = WordDataset(["cat", "car"], tokenizer)
xs, ys = dataset.training_data()  # Returns pairs of consecutive characters
```

## Motivation

The dual implementation approach demonstrates different ways to solve the same problem:

1. **Statistical Model**: 
   - Simple and interpretable
   
2. **Neural Model**:
   - Better generalization potential

## Usage

Basic example:
```python
words = ["cat", "cab", "can", "car"]
tokenizer = TokenizerCharacter(words)
dataset = WordDataset(words, tokenizer)

# Statistical approach
markov = TransitionMatrix.from_words(words, tokenizer)
print(markov.generate_sequence())

# Neural approach
neural = NeuralTransitionMatrix(tokenizer)
neural.train_bigram_model(dataset)
print(neural.generate_sequence())
```
