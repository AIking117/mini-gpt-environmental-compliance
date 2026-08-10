# MiniGPT for Environmental Compliance

Potential Name: DEPGPT

This repository is an ongoing learning project focused on building a small GPT-style language model from scratch in Python and PyTorch.

The first goal is to understand the core components of a decoder-only transformer language model, including tokenization, embeddings, attention, transformer blocks, training, and text generation.

The longer-term goal is to explore how this type of model could be adapted toward environmental compliance and infrastructure-related language, including MS4, MSGP, SWPPP, stormwater inspection, and corrective action workflows.

## Project Goals

1. Build a small GPT-style model from scratch.
2. Learn the internal mechanics of language models step by step.
3. Strengthen PyTorch and machine learning engineering skills through hands-on implementation.
4. Explore how language models can support environmental compliance and infrastructure workflows.

## Current Status

Completed:
- Project repository initialized
- Basic folder structure created
- Character-level tokenizer implemented

In progress:
- File-based text input
- Dataset preparation
- Token batching for next-token prediction

Planned:
- Token embeddings
- Positional embeddings
- Self-attention
- Multi-head attention
- Transformer block
- Training loop
- Text generation
- Domain-specific tokenizer experiments
- Environmental compliance fine-tuning prototype

## Repository Structure

```text
mini-gpt-environmental-compliance/
│
├── data/
│   └── raw/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
│
├── README.md
└── requirements.txt# Dataset Utilities
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
```

## Learning Notes: Dataset Preparation                         # Created: 2026-06-08

The current dataset stage does not train the model yet. It prepares training examples for future model training.

The tokenizer converts raw text into token IDs. Then `dataset.py` takes those token IDs and creates input/target pairs for next-token prediction.

Example with a character-level tokenizer and `context_length = 16`:

Input:

    Stormwater inspec

Target:

    tormwater inspect

The input is what the future model will see. The target is the correct shifted sequence that the model should learn to predict.

This is one training example, not one completed learning step. Actual learning will happen later when a model makes predictions, compares those predictions to the targets, calculates loss, and updates model weights.

The dataset creation process moves left to right through the text using a sliding window:

    Example 1: tokens 0–15 → target tokens 1–16
    Example 2: tokens 1–16 → target tokens 2–17
    Example 3: tokens 2–17 → target tokens 3–18

`context_length` controls how many tokens are included in each input sequence. A smaller context length is simpler and cheaper, but gives the model less context. A larger context length gives the model more context, but requires more memory and computation.

At this stage, no model weights are being updated. `dataset.py` only prepares the practice material for the future MiniGPT model.