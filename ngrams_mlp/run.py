from ngrams_mlp.chargen import CharacterGenerator
from ngrams_mlp.word_dataset import WordDataset
from bigrams.tokenizer import TokenizerCharacter
from ngrams_mlp.ngram_nn import NGramNN

def _read_names():
    with open('../names.txt') as f:
        return f.read().splitlines()


if __name__ == "__main__":
    # words = ['hello', 'world', 'neural', 'network', 'test']
    words = _read_names()
    tokenizer = TokenizerCharacter(words)

    # Create and train model
    model = NGramNN(tokenizer=tokenizer, block_size=2)  # block_size=2 for bigram
    dataset = WordDataset(words, tokenizer)
    model.train(dataset, num_epochs=100)

    # Initialize generator
    generator = CharacterGenerator(model, tokenizer)

    # Example 1: Predict next character
    context = "ne"
    next_char = generator.predict_next(context)
    print(f"After '{context}' predicts: {next_char}")

    # Example 2: Calculate probability of specific sequence
    prob = generator.sequence_probability("ne", "l")
    print(f"P(l|ne) = {prob:.3f}")

    # Example 3: Find top 3 likely next characters
    predictions = generator.top_k_predictions("ne", k=3)
    print("Top 3 predictions after 'ne':")
    for char, prob in predictions:
        print(f"  {char}: {prob:.3f}")

    # Example 4: Generate some words
    sampled_words = generator.sample_words(10)
    print("\nSampled words:")
    for word in sampled_words:
        print(word)