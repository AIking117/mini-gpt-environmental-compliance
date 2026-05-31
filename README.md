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