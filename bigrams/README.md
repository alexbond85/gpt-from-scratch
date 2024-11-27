# Learning vs Computing Character Transition Matrices for Word Generation

This project demonstrates two approaches to obtain a character transition matrix for generating synthetic words:
1. Direct computation from data (statistical counting)
2. Learning through gradient descent

## The Core Idea: Transition Matrix

Both approaches aim to create the same thing: a matrix T where T[i,j] represents P(char_j | char_i). For example, with words ["cat", "car"]:

```
Ideal transition matrix (simplified):
     c    a    t    r   END
START 1.0  0    0    0    0
c     0   1.0  0    0    0
a     0    0   0.5  0.5  0
t     0    0    0    0   1.0
r     0    0    0    0   1.0
```

## Approach 1: Direct Computation

Count transitions and normalize:
```python
# For "cat", "car":
transitions = {
    ('START', 'c'): 2,
    ('c', 'a'): 2,
    ('a', 't'): 1,
    ('a', 'r'): 1,
    ('t', 'END'): 1,
    ('r', 'END'): 1
}

# Normalize per character:
P(next|current) = count(current->next) / sum(counts from current)
```

## Approach 2: Learning through Gradient Descent

Instead of counting, we:
1. Initialize random matrix W
2. Convert characters to one-hot vectors
3. Predict next character: P(next) = softmax(current @ W)
4. Update W to minimize cross-entropy loss

Example training step:
```python
# For transition START->c in "cat":
current = [1,0,0,0,0]  # START one-hot
target = [0,1,0,0,0]   # 'c' one-hot

pred = softmax(current @ W)
loss = -log(pred[target])
W = W - learning_rate * gradient
```

## Key Differences

1. Statistical:
   - Exact counts from data
   - Zero probability for unseen transitions
   - Training time: O(N) where N is dataset size
   
2. Learning:
   - Smooth probabilities through optimization
   - Non-zero probabilities for unseen transitions (due to softmax)
   - Training time: O(N * epochs)

## Evaluation

Both methods should converge to similar matrices if:
1. Learning rate and epochs are sufficient
2. Dataset is large enough
3. No regularization is applied

The minimum achievable loss is the same for both, as they model the same underlying probability distribution.
