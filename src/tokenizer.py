# ==========================================================
# Character Tokenizer
#
# Purpose:
# Convert text into token IDs and convert token IDs back
# into text.
#
# This is the first building block of an LLM.
#
# Example:
#
# "SWPPP"
#
# becomes:
#
# [5, 8, 3, 3, 3]
#
# and can later be converted back into:
#
# "SWPPP"
#
# ==========================================================


class CharacterTokenizer:

    # ------------------------------------------------------
    # Constructor
    #
    # Builds the vocabulary and lookup dictionaries.
    # ------------------------------------------------------
    def __init__(self, text):

        # Get all unique characters from the text
        self.chars = sorted(list(set(text)))

        # Number of unique characters
        self.vocab_size = len(self.chars)

        # String-to-Integer dictionary
        #
        # Example:
        # {'S': 0, 'W': 1, 'P': 2}
        #
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

        # Integer-to-String dictionary
        #
        # Example:
        # {0: 'S', 1: 'W', 2: 'P'}
        #
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    # ------------------------------------------------------
    # Encode
    #
    # Converts text into token IDs.
    #
    # Example:
    # "SWPPP"
    #
    # becomes:
    # [5, 8, 3, 3, 3]
    # ------------------------------------------------------
    def encode(self, text):

        return [self.stoi[ch] for ch in text]

    # ------------------------------------------------------
    # Decode
    #
    # Converts token IDs back into text.
    #
    # Example:
    # [5, 8, 3, 3, 3]
    #
    # becomes:
    # "SWPPP"
    # ------------------------------------------------------
    def decode(self, ids):

        return "".join([self.itos[i] for i in ids])


# ==========================================================
# Main Program
#
# This section runs only when we execute:
#
# python src/tokenizer.py
#
# ==========================================================
if __name__ == "__main__":        # Below code runs only when we execute: python src/tokenizer.py

    # Sample text for testing
    sample_text = "SWPPP and MS4 compliance"

    # Create tokenizer object
    tokenizer = CharacterTokenizer(sample_text)

    # Encode text into token IDs
    encoded = tokenizer.encode(sample_text)

    # Decode token IDs back into text
    decoded = tokenizer.decode(encoded)

    # Display results
    print("Original text:", sample_text)
    print("Vocabulary:", tokenizer.chars)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Encoded:", encoded)
    print("Decoded:", decoded)