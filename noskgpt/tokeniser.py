""" 
Character-level tokeniser.
"""

import torch


class CharToIntTokeniser:
    """
    Simple tokeniser which represents characters as integers.
    """

    def __init__(self, corpus: str) -> None:
        """
        Create the encoder and decoder mappings from a corpus of text.

        :param corpus: Corpus from which to create the encoder mapping.
        """
        chars = sorted(list(set(corpus)))
        self.char_to_int = {c: i for i, c in enumerate(chars)}
        self.int_to_char = {i: c for i, c in enumerate(chars)}

    def encode(self, raw_text: str) -> torch.tensor:
        """
        Encode text using character-to-integer mapping.

        :param raw_text: Raw text.
        :return: Encoded text.
        """
        encoded_chars = [self.char_to_int[c] for c in raw_text]
        return torch.tensor(encoded_chars, dtype=torch.long).view(-1)

    def decode(self, encoded_text: torch.tensor) -> str:
        """
        Decode encoded text using integer-to-character mapping.

        :param encoded_text: Encoded text.
        :return: Decoded text.
        """
        chars = [self.int_to_char[i] for i in encoded_text.view(-1).tolist()]
        return "".join(chars)

