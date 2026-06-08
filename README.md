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
└── requirements.txt
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