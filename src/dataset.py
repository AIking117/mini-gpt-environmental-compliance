# Dataset Utilities
# Purpose:
# Convert token IDs into input/target pairs for next-token prediction.

import torch
from tokenizer import CharacterTokenizer


def create_input_target_pairs(token_ids, context_length):
    """
    Create input and target sequences for next-token prediction.

    token_ids:
        A list of integer token IDs.

    context_length:
        How many token IDs the model sees at one time.

    Example:
        token_ids = [10, 20, 30, 40, 50]
        context_length = 4

        input  = [10, 20, 30, 40]
        target = [20, 30, 40, 50]
    """

    inputs = []
    targets = []

    for i in range(len(token_ids) - context_length):
        x = token_ids[i : i + context_length]
        y = token_ids[i + 1 : i + context_length + 1]

        inputs.append(x)
        targets.append(y)
                  # Converting Python lists to PyTorch tensors. Each row is a training example. Each column is one token positionm inside the context window.
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)    
    targets_tensor = torch.tensor(targets, dtype=torch.long)  

    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    file_path = "data/raw/sample.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        sample_text = file.read()

    tokenizer = CharacterTokenizer(sample_text)
    token_ids = tokenizer.encode(sample_text)

    context_length = 16

    inputs, targets = create_input_target_pairs(token_ids, context_length)

    print("File path:", file_path)
    print("Total characters:", len(sample_text))
    print("Total token IDs:", len(token_ids))
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Context length:", context_length)
    print("Number of training examples:", len(inputs))
    print("Input tensor shape:", inputs.shape)
    print("Target tensor shape:", targets.shape)

    print("\nFirst input token IDs:")
    print(inputs[0])

    print("\nFirst target token IDs:")
    print(targets[0])

    # Convert the tensor row back to a Python list so the tokenizer can decode it.
    # The tokenizer's decode method expects regular integer IDs, not tensor values.
    print(tokenizer.decode(inputs[0].tolist()))
    print(tokenizer.decode(targets[0].tolist()))