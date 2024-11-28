from typing import List, Dict


class TokenizerCharacter:
    """
    A character-level tokenizer that converts between characters and their corresponding indices.
    Handles special tokens for start ('<S>') and end ('<E>') of sequences, represented as '.'
    in the encoded space.

    Attributes:
        stoi (Dict[str, int]): String-to-index mapping for encoding characters to integers
        itos (Dict[int, str]): Index-to-string mapping for decoding integers back to characters
        vocab_size (int): Size of the vocabulary (number of unique characters + special tokens)
    """

    def __init__(self, words: List[str]) -> None:
        """
        Initialize the tokenizer with a list of words to build the vocabulary.

        Args:
            words: List of words to build the vocabulary from
                  Example: ["cat", "dog"]
        """
        # Extract unique characters from words and sort them for consistency
        chars = sorted(list(set(''.join(words))))

        # Create character to index mapping (stoi)
        # Reserve index 0 for the special marker '.' (represents both <S> and <E>)
        self.stoi: Dict[str, int] = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0  # Special token

        # Create index to character mapping (itos)
        self.itos: Dict[int, str] = {i: s for s, i in self.stoi.items()}

        # Store vocabulary size
        self._vocab_size: int = len(self.stoi)

    @property
    def vocab_size(self) -> int:
        """
        Size of the vocabulary.

        Returns:
            Number of unique tokens including special tokens
        """
        return self._vocab_size

    def encode_char(self, char: str) -> int:
        """
        Encode a single character to its corresponding index.

        Args:
            char: Character to encode
                 Example: 'a'

        Returns:
            Index corresponding to the character
            Example: 1
        """
        return self.stoi.get(char, self.stoi['.'])  # Default to special token if char not found

    def decode_char(self, index: int) -> str:
        """
        Decode a single index back to its corresponding character.

        Args:
            index: Index to decode
                  Example: 1

        Returns:
            Character corresponding to the index
            Example: 'a'
        """
        return self.itos.get(index, '.')  # Default to special token if index not found

    def encode_word(self, word: str) -> List[int]:
        """
        Encode a complete word to a list of indices, adding start and end tokens.

        Args:
            word: Word to encode
                 Example: "cat"

        Returns:
            List of indices including start and end tokens
            Example: [0, 3, 1, 20, 0]  # [<S>, 'c', 'a', 't', <E>]
        """
        chars = ['.'] + list(word) + ['.']  # Add start/end markers
        return [self.encode_char(c) for c in chars]

    def decode_indices(self, indices: List[int]) -> str:
        """
        Decode a list of indices back to a word, removing special tokens.

        Args:
            indices: List of indices to decode
                    Example: [0, 3, 1, 20, 0]

        Returns:
            Decoded word without special tokens
            Example: "cat"
        """
        chars = [self.decode_char(idx) for idx in indices]
        # Remove special tokens and join characters
        return ''.join(c for c in chars if c != '.')