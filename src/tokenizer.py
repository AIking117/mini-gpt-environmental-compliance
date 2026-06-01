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
if __name__ == "__main__":

    # Path to the raw text file
    file_path = "data/raw/sample.txt"

    # Read the full text file into Python as one string
    with open(file_path, "r", encoding="utf-8") as file:
        sample_text = file.read()

    # Create tokenizer object using text from the file
    tokenizer = CharacterTokenizer(sample_text)

    # Encode text into token IDs
    encoded = tokenizer.encode(sample_text)

    # Decode token IDs back into text
    decoded = tokenizer.decode(encoded)

    # Display results
    print("File path:", file_path)
    print("Original text preview:", sample_text[:120])
    print("Vocabulary:", tokenizer.chars)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Encoded preview:", encoded[:60])
    print("Decoded preview:", decoded[:120])