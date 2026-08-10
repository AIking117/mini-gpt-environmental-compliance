# Bedrock

Bedrock is an educational implementation of a GPT-style decoder-only transformer built from scratch in Python and PyTorch.

The project follows the concepts and implementation approach in Sebastian Raschka's *Build a Large Language Model (From Scratch)* and is intended to develop a practical understanding of how modern language models work internally.

## Purpose

The goal of Bedrock is not to produce a production-ready chatbot.

The goal is to understand the major components of a GPT-style language model by implementing them step by step, including:

* Tokenization
* Dataset preparation
* Next-token prediction
* Token embeddings
* Positional embeddings
* Masked self-attention
* Multi-head attention
* Transformer blocks
* Training and loss calculation
* Text generation

The project emphasizes understanding the mechanics of the model rather than relying on high-level APIs or pretrained language models.

## Project Pipeline

```
Raw Text
    ↓
Tokenization
    ↓
Input / Target Training Examples
    ↓
PyTorch Tensors
    ↓
Token Embeddings
    ↓
Positional Embeddings
    ↓
Masked Self-Attention
    ↓
Multi-Head Attention
    ↓
Transformer Blocks
    ↓
Training
    ↓
Text Generation
```

## Current Status

Completed:

* Project structure
* Character-level tokenizer
* Text encoding and decoding
* File-based text input
* Sliding-window input/target generation
* Configurable context length
* PyTorch tensor conversion

Next:

* Token embeddings
* Positional embeddings

Planned:

* Masked self-attention
* Multi-head attention
* Transformer blocks
* Training loop
* Loss calculation
* Text generation
* Model evaluation

## Learning Notes: Dataset Preparation

The dataset stage does not train the model. It prepares examples that will later be used during model training.

The tokenizer converts raw text into token IDs. `dataset.py` then creates input and target sequences for next-token prediction.

For example, using a character-level tokenizer with a context length of 16:

```
Input:
Stormwater inspe

Target:
tormwater inspec
```

The input is the sequence presented to the model.

The target is the correct sequence shifted one token forward.

This represents one training example. Learning does not occur until a model produces predictions, those predictions are compared with the targets, a loss is calculated, and the model weights are updated.

A sliding window creates many training examples:

```
Example 1: tokens 0–15 → target tokens 1–16
Example 2: tokens 1–16 → target tokens 2–17
Example 3: tokens 2–17 → target tokens 3–18
```

The context length controls the number of tokens included in each input sequence.

## Relationship to Stormwater RAG

Bedrock and Stormwater RAG are separate projects.

Bedrock answers:

> How does a GPT-style language model work internally?

Stormwater RAG answers:

> How can existing language-model technology be used to retrieve and explain stormwater and environmental-compliance information?

Stormwater RAG does not use the model built in Bedrock as its language-model engine.

The two projects were intentionally separated because building an LLM from scratch for education and building a practical retrieval-augmented application are different engineering objectives.

Knowledge developed through Bedrock may inform future AI engineering decisions, but neither project depends on the other.
