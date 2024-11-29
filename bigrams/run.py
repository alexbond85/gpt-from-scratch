from bigrams.loss import calculate_min_loss
from bigrams.tokenizer import TokenizerCharacter
from bigrams.trmatrix import TransitionMatrix
from bigrams.trmatrix_nn import NeuralTransitionMatrix
from bigrams.viz import print_matrix_section
from bigrams.word_dataset import WordDataset


def _read_names():
    with open('../names.txt') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    # Example usage showing training and generation
    words = ["cat", "cab", "can", "car", "bat", "bar", "ban"]
    # words = _read_names()
    tokenizer = TokenizerCharacter(words)
    words_dataset = WordDataset(words, tokenizer)
    markov_matrix = TransitionMatrix.from_words(words, tokenizer)
    print("\nTransition probabilities:")
    print_matrix_section(markov_matrix, tokenizer)
    for i in range(5):
        print(markov_matrix.generate_sequence(seed=i))
    print("Min loss:", calculate_min_loss(markov_matrix.probabilities, *words_dataset.training_data()))
    matrix = NeuralTransitionMatrix(tokenizer)
    matrix.train_bigram_model(words_dataset)
    print_matrix_section(matrix, tokenizer)
    print("\nGenerated words:")
    for i in range(5):
        print(matrix.generate_sequence(seed=i))