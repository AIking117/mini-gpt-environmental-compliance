# ==========================================================
# Dataset Utilities                       # Created: 2026-06-01
#
# Purpose:
# Convert token IDs into input/target pairs for next-token
# prediction.
#
# GPT-style models learn by predicting the next token.
#
# Example:
#
# tokens: [10, 20, 30, 40, 50]
#
# input:  [10, 20, 30, 40]
# target: [20, 30, 40, 50]
#
# ==========================================================


def create_input_target_pairs(token_ids, context_length):
    """
    Create training examples for next-token prediction.

    Parameters
    ----------
    token_ids : list[int]
        A list of token IDs created by the tokenizer.

    context_length : int
        How many tokens the model sees at one time.

    Returns
    -------
    inputs : list[list[int]]
        Input token sequences.

    targets : list[list[int]]
        Target token sequences shifted one step forward.
    """

    inputs = []
    targets = []

    # We stop at len(token_ids) - context_length
    # because each target needs one extra token after the input.
    for i in range(len(token_ids) - context_length):

        # Input sequence
        x = token_ids[i : i + context_length]

        # Target sequence shifted one token forward
        y = token_ids[i + 1 : i + context_length + 1]

        inputs.append(x)
        targets.append(y)

    return inputs, targets


# ==========================================================
# Main Program
#
# This section is only for testing this file directly.
#
# Run:
#
# python src/dataset.py
#
# ==========================================================
if __name__ == "__main__":

    # Small fake token list for testing
    token_ids = [10, 20, 30, 40, 50, 60]

    # The model will see 4 tokens at a time
    context_length = 4

    inputs, targets = create_input_target_pairs(token_ids, context_length)

    print("Token IDs:", token_ids)
    print("Context length:", context_length)
    print()

    for i in range(len(inputs)):
        print(f"Example {i + 1}")
        print("Input: ", inputs[i])
        print("Target:", targets[i])
        print()
