import torch
import matplotlib.pyplot as plt


def read_words(filename):
    """
    Reads words from a file and splits them into a list, with each word on a new line.

    Args:
        filename (str): Path to the file containing words, where each word is on a separate line.
                       Example: "names.txt" containing ["John", "Mary", "Peter"]

    Returns:
        list: List of words from the file, with whitespace and newlines removed.
              Example: For file containing "John\nMary\nPeter", returns ["John", "Mary", "Peter"]

    Example Usage:
    -------------
    words = read_words('names.txt')
    # If names.txt contains:
    # John
    # Mary
    # Peter
    # Then words will be ['John', 'Mary', 'Peter']
    """
    with open(filename, 'r') as file:
        return file.read().splitlines()


def compute_bigram_counts(words):
    """
    Compute counts for each bigram (pairs of consecutive characters) in the words,
    including special start '<S>' and end '<E>' markers.

    Args:
        words (list): List of words to analyze.
                     Example: ["cat", "dog"]

    Returns:
        dict: Dictionary with bigram tuples as keys and their counts as values.
              Each bigram includes the transition from one character to the next.
              Example: For ["cat"], returns:
              {('<S>', 'c'): 1, ('c', 'a'): 1, ('a', 't'): 1, ('t', '<E>'): 1}

    Example Workflow:
    ----------------
    1. For each word, add start and end markers:
       "cat" -> ['<S>', 'c', 'a', 't', '<E>']
    2. Count transitions between consecutive characters:
       ('<S>', 'c'), ('c', 'a'), ('a', 't'), ('t', '<E>')
    3. Accumulate counts across all words in the dictionary
    """
    bigram_counts = {}
    for word in words:
        chars = ['<S>'] + list(word) + ['<E>']  # Add start and end markers
        for ch1, ch2 in zip(chars, chars[1:]):
            bigram = (ch1, ch2)
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    return bigram_counts


def sort_bigram_counts(bigram_counts):
    """
    Sort bigram counts in descending order of their frequency to identify
    the most common character transitions.

    Args:
        bigram_counts (dict): Dictionary containing bigram counts where:
                             - Keys are tuples of character pairs: ('a', 'b')
                             - Values are integers representing counts
                             Example: {('t', 'h'): 50, ('e', 'r'): 30}

    Returns:
        list: List of tuples sorted by count in descending order, where each tuple contains:
              ((char1, char2), count)
              Example: [(('t', 'h'), 50), (('e', 'r'), 30)]

    Example Usage:
    -------------
    bigrams = {('t', 'h'): 50, ('e', 'r'): 30}
    sorted_bigrams = sort_bigram_counts(bigrams)
    # Returns: [(('t', 'h'), 50), (('e', 'r'), 30)]
    """
    return sorted(bigram_counts.items(), key=lambda kv: -kv[1])


def initialize_transition_matrix(words, alphabet_size=27):
    """
    Initialize and compute the bigram transition matrix for characters.

    Args:
        words (list): List of words. Example: ["cat", "dog"]
        alphabet_size (int): Size of the alphabet (+1 for start/end markers).
                             Example: If we have 26 letters (a-z), add 1 for the marker.

    Returns:
        tuple:
            - transition_matrix (torch.Tensor): A matrix where entry [i, j] represents
              the frequency of transitions from character i to character j.
            - stoi (dict): A dictionary mapping characters to indices. Example: {'a': 1, 'b': 2, ..., '.': 0}
            - itos (dict): A dictionary mapping indices to characters. Example: {0: '.', 1: 'a', 2: 'b', ...}

    Explanation of Steps:
    ---------------------
    This function creates a bigram transition matrix that records how often one character transitions to another.
    For example, in the word "cat", the transitions are:
        - Start ('<S>') -> 'c'
        - 'c' -> 'a'
        - 'a' -> 't'
        - 't' -> End ('<E>')

    These transitions are captured in a 2D matrix where:
        - Rows correspond to "from" characters.
        - Columns correspond to "to" characters.
    The matrix is updated by counting these transitions for every word in the dataset.

    Example Workflow:
    -----------------
    Input words: ["cat", "dog"]

    Step 1: Create a list of all unique characters:
        - Unique characters: ['c', 'a', 't', 'd', 'o', 'g']
        - Add start/end marker: '.' (for '<S>' and '<E>')

    Step 2: Map these characters to indices:
        - stoi: {'c': 1, 'a': 2, 't': 3, 'd': 4, 'o': 5, 'g': 6, '.': 0}
        - itos: {1: 'c', 2: 'a', 3: 't', 4: 'd', 5: 'o', 6: 'g', 0: '.'}

    Step 3: Create a 2D matrix initialized to zeros:
        - Size: 7 x 7 (for 7 characters including the marker)

    Step 4: Process each word and update the matrix:
        - For "cat": ['.', 'c', 'a', 't', '.']
          Transitions: ('.' -> 'c'), ('c' -> 'a'), ('a' -> 't'), ('t' -> '.')
          Matrix updates these transitions' counts.

        - For "dog": ['.', 'd', 'o', 'g', '.']
          Transitions: ('.' -> 'd'), ('d' -> 'o'), ('o' -> 'g'), ('g' -> '.')
          Matrix updates these transitions' counts.

    Result:
        Transition matrix for "cat" and "dog":
           .  c  a  t  d  o  g
        .   0  1  0  0  1  0  0
        c   0  0  1  0  0  0  0
        a   0  0  0  1  0  0  0
        t   1  0  0  0  0  0  0
        d   0  0  0  0  0  1  0
        o   0  0  0  0  0  0  1
        g   1  0  0  0  0  0  0
    """

    # Step 1: Extract all unique characters from the dataset
    # Collect all unique characters from the words and sort them for consistency
    chars = sorted(list(set(''.join(words))))

    # Add a special marker '.' to represent the start ('<S>') and end ('<E>') of words
    # Create a mapping from each character to a unique index (stoi: string-to-index)
    # Assign the index 0 to the special marker '.'
    stoi = {s: i + 1 for i, s in enumerate(chars)}  # Characters get indices starting from 1
    stoi['.'] = 0  # Marker '.' gets index 0

    # Create a reverse mapping from indices to characters (itos: index-to-string)
    itos = {i: s for s, i in stoi.items()}

    # Step 2: Initialize the transition matrix
    # This is a square matrix of size alphabet_size x alphabet_size, initialized to zeros
    # Each entry [i, j] will hold the count of transitions from character i to character j
    transition_matrix = torch.zeros((alphabet_size, alphabet_size), dtype=torch.int32)

    # Step 3: Process each word in the dataset to update the transition matrix
    for word in words:
        # Add the start and end markers to the word
        # For example, "cat" becomes ['.', 'c', 'a', 't', '.']
        chars = ['.'] + list(word) + ['.']

        # Iterate through consecutive character pairs using zip
        for ch1, ch2 in zip(chars, chars[1:]):
            # Convert the characters to their indices using stoi mapping
            ix1 = stoi[ch1]  # Index for the "from" character
            ix2 = stoi[ch2]  # Index for the "to" character

            # Increment the corresponding cell in the transition matrix
            # This counts the transition from ch1 to ch2
            transition_matrix[ix1, ix2] += 1

    # Step 4: Return the transition matrix and the mappings for further analysis or visualization
    return transition_matrix, stoi, itos


def normalize_transition_probabilities(matrix, row_index):
    """
    Normalizes a specific row in the transition matrix to create a valid probability
    distribution where all values sum to 1.0.

    Args:
        matrix (torch.Tensor): The transition matrix where each cell [i,j] represents
                             the count of transitions from character i to character j.
        row_index (int): Index of the row to normalize. Each row represents transitions
                        from a specific character.

    Returns:
        torch.Tensor: A normalized probability distribution where all values are between
                     0 and 1, and sum to 1.0.

    Example:
    --------
    If row contains counts [2, 3, 5]:
    1. Convert to float: [2.0, 3.0, 5.0]
    2. Sum = 10.0
    3. Normalize: [0.2, 0.3, 0.5]
    """
    row = matrix[row_index].float()  # Get the row and convert to float for division
    probabilities = row / row.sum()  # Normalize to sum to 1
    return probabilities


def sample_character(probabilities, generator, itos):
    """
    Samples the next character based on the given probability distribution
    using the provided random number generator.

    Args:
        probabilities (torch.Tensor): A tensor of probabilities for each possible
                                    next character, summing to 1.0.
        generator (torch.Generator): A torch random number generator for reproducible
                                   sampling.
        itos (dict): Index-to-string mapping dictionary that converts numeric indices
                     to their corresponding characters.
                     Example: {0: '.', 1: 'a', 2: 'b'}

    Returns:
        str: The sampled character based on the probability distribution.
             Example: If probabilities are [0.2, 0.5, 0.3] for ['a', 'b', 'c'],
             it might return 'b' with 50% probability.
    """
    # Sample an index based on the probabilities
    sampled_index = torch.multinomial(probabilities, num_samples=1, replacement=True, generator=generator).item()
    return itos[sampled_index]


def analyze_and_sample_character(transition_matrix, itos, row_index=0, seed=2147483647):
    """
    Wrapper function that combines normalization of transition probabilities
    and character sampling to generate the next character in a sequence.

    Args:
        transition_matrix (torch.Tensor): Matrix of transition counts between characters.
        itos (dict): Index-to-string mapping for converting indices to characters.
                     Example: {0: '.', 1: 'a', 2: 'b'}
        row_index (int): Index of the current character in the transition matrix.
                        Default is 0 (start marker).
        seed (int): Random seed for reproducible sampling. Default is 2147483647.

    Returns:
        str: A single character sampled based on the transition probabilities
             from the current character (row_index).

    Example Usage:
    -------------
    # Starting from the beginning of a word (row_index=0)
    first_char = analyze_and_sample_character(matrix, itos)
    # This might return 't' if 't' is a common starting character
    """
    probabilities = normalize_transition_probabilities(transition_matrix, row_index)
    generator = torch.Generator().manual_seed(seed)  # Set random seed for reproducibility
    sampled_character = sample_character(probabilities, generator, itos)
    return sampled_character


def generate_word(transition_matrix, itos, seed=2147483647, max_length=50):
    """
    Generates a new word by sampling characters based on transition probabilities
    until an end marker is reached or maximum length is exceeded.

    Args:
        transition_matrix (torch.Tensor): Matrix of transition probabilities between characters.
        itos (dict): Index-to-string mapping for converting indices to characters.
                     Example: {0: '.', 1: 'a', 2: 'b'}
        seed (int): Random seed for reproducible generation. Default is 2147483647.
        max_length (int): Maximum length of generated word to prevent infinite loops.
                         Default is 50 characters.

    Returns:
        str: A generated word that follows the transition patterns of the training data.
             Example: "michael" or "jennifer"

    Example Workflow:
    ----------------
    1. Start with marker (index 0)
    2. Sample next character based on transition probabilities
    3. Continue sampling until end marker is reached or max_length
    4. Join characters to form final word

    Example Usage:
    -------------
    word = generate_word(matrix, itos, seed=42)
    # Might return "michael" if trained on names
    """
    g = torch.Generator().manual_seed(seed)  # Initialize random generator with fixed seed
    word = []  # Store sampled characters here
    current_index = 0  # Start sampling from the start marker ('<S>')

    for _ in range(max_length):
        # Get the probability distribution for the current character
        probabilities = transition_matrix[current_index].float()  # Convert to float for multinomial

        # Normalize probabilities to ensure they sum to 1
        probabilities = probabilities / probabilities.sum()

        # Sample the next character index
        next_index = torch.multinomial(probabilities, num_samples=1, replacement=True, generator=g).item()

        # Append the sampled character to the word
        word.append(itos[next_index])

        # Stop if the end marker ('<E>') is reached
        if next_index == 0:
            break

        # Move to the next character index
        current_index = next_index

    # Join the sampled characters into a single string (without the end marker)
    return ''.join(word)  # Exclude the '<E>' marker


def visualize_transition_matrix(matrix, itos):
    """
    Creates a visual representation of the transition matrix as a heatmap,
    showing character transition patterns with both characters and counts.

    Args:
        matrix (torch.Tensor): The transition matrix to visualize, where each cell [i,j]
                             represents the count or probability of transitioning from
                             character i to character j.
        itos (dict): Index-to-string mapping for converting indices to characters.
                     Example: {0: '.', 1: 'a', 2: 'b'}

    Visual Output:
    -------------
    - Creates a figure with a blue heatmap where:
      - Darker colors indicate higher transition probabilities
      - Each cell shows:
        - Top: The character pair (e.g., "ab" for transition from 'a' to 'b')
        - Bottom: The transition count or probability value

    Example Usage:
    -------------
    visualize_transition_matrix(transition_matrix, itos)
    # Displays a heatmap showing how likely each character is to follow another
    """
    plt.figure(figsize=(16, 16))
    plt.imshow(matrix, cmap='Blues')
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            chstr = itos.get(i, '') + itos.get(j, '')
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')  # Character pair
            plt.text(j, i, round(matrix[i, j].item(), 2), ha="center", va="top", color='gray')  # Count value
    plt.axis('off')
    plt.show()

def normalize_transition_matrix(matrix):
    """
    Normalizes the entire transition matrix so that each row represents a valid
    probability distribution summing to 1.0.

    Args:
        matrix (torch.Tensor): The raw transition count matrix to be normalized.
                             Shape: [n_chars, n_chars] where n_chars is the number
                             of unique characters plus special markers.

    Returns:
        torch.Tensor: Normalized transition matrix where each row sums to 1.0,
                     representing valid probability distributions for character
                     transitions.

    Example Workflow:
    ----------------
    1. Convert matrix to floating point for division
    2. Sum each row while keeping dimensions for broadcasting
    3. Handle zero rows to prevent division by zero
    4. Divide each row by its sum

    Example:
    --------
    Raw matrix row: [2, 3, 5]
    Normalized row: [0.2, 0.3, 0.5]
    """
    matrix = matrix.float()  # Ensure floating-point values
    row_sums = matrix.sum(dim=1, keepdim=True)  # Sum of each row
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return matrix / row_sums


def calculate_logloss(probs: torch.Tensor, true_indices: torch.Tensor) -> float:
    """
    Calculate the negative log likelihood loss for predicted probabilities.

    Args:
        probs (torch.Tensor): Predicted probabilities for each character [batch_size, vocab_size]
        true_indices (torch.Tensor): True character indices [batch_size]

    Returns:
        float: Average negative log likelihood loss
    """
    # Get probabilities for the true characters
    true_probs = probs[torch.arange(len(true_indices)), true_indices]
    # Calculate negative log likelihood
    nll = -torch.log(true_probs)
    return nll.mean().item()


def main():
    """
    Main function that demonstrates the complete workflow of analyzing character patterns
    in words and generating new words based on these patterns. This process follows the
    principles of Markov chains, where each character's probability depends only on the
    previous character.

    The workflow consists of several key steps:
    1. Data Loading: Read training words from file
    2. Pattern Analysis: Analyze character transition patterns
    3. Matrix Creation: Build and normalize a transition probability matrix
    4. Word Generation: Use the learned patterns to generate new words

    This implementation is particularly useful for:
    - Name generation for games or creative writing
    - Understanding pattern recognition in natural language
    - Demonstrating practical applications of Markov chains
    """
    print("=== Character-based Markov Chain Word Generator ===\n")

    # Step 1: Load words from file
    print("Step 1: Loading training data...")
    words = read_words('../names.txt')
    print(f"Loaded {len(words)} words for training")
    print(f"Sample words from dataset: {', '.join(words[:5])}\n")

    # Step 2: Compute bigram counts to understand character patterns
    print("Step 2: Analyzing character patterns...")
    bigram_counts = compute_bigram_counts(words)

    # Show most common character transitions for insight into the patterns
    sorted_counts = sort_bigram_counts(bigram_counts)
    print("Most frequent character transitions:")
    for (ch1, ch2), count in sorted_counts[:10]:
        # Convert special markers for clearer display
        ch1_display = 'START' if ch1 == '<S>' else ('END' if ch1 == '<E>' else ch1)
        ch2_display = 'START' if ch2 == '<S>' else ('END' if ch2 == '<E>' else ch2)
        print(f"  '{ch1_display}' → '{ch2_display}': {count} times")
    print()

    # Step 3: Initialize and normalize the transition matrix
    print("Step 3: Building transition probability matrix...")
    transition_matrix, stoi, itos = initialize_transition_matrix(words)

    # Display matrix size and character set
    print(f"Matrix size: {transition_matrix.shape[0]}x{transition_matrix.shape[1]}")
    print(f"Character set: {', '.join(sorted(stoi.keys()))}\n")

    # Normalize the matrix to get probabilities
    transition_matrix = normalize_transition_matrix(transition_matrix)

    # Step 4: Demonstrate character sampling
    print("Step 4: Testing character sampling...")
    print("Sampling first characters of words (after START marker):")
    for _ in range(5):
        char = analyze_and_sample_character(transition_matrix, itos, row_index=0, seed=torch.seed())
        print(f"  Sampled starting character: '{char}'")
    print()

    # Step 5: Generate complete words
    print("Step 5: Generating new words using learned patterns...")
    print("Generated words (using different random seeds):")
    for i in range(10):  # Generate 10 words for a good demonstration
        # Use varying seeds for diversity
        generated_word = generate_word(transition_matrix, itos, seed=torch.seed())
        # Remove the '.' character that represents the end marker
        cleaned_word = generated_word.replace('.', '')
        print(f"  Word {i + 1}: {cleaned_word}")

    # Optional: Visualize the transition matrix
    print("\nStep 6: Visualizing transition patterns...")
    print("Displaying transition matrix heatmap...")
    # visualize_transition_matrix(transition_matrix, itos)
    print("Visualization complete. Darker colors indicate more frequent transitions.")


if __name__ == '__main__':
    main()