# gpt2_shakespeare: Custom GPT-2 Transformer Decoder on Tiny Shakespeare

This repository contains a custom implementation of a GPT‑2–style model using PyTorch. Instead of relying on PyTorch's built-in Transformer modules, this code defines its own Transformer Decoder blocks with masked self-attention, and it is trained on the Tiny Shakespeare dataset.

## Overview

This project demonstrates how to:
- Build a character-level vocabulary from the Tiny Shakespeare text.
- Implement custom Multi-Head Self-Attention and Transformer Decoder Blocks.
- Create a GPT‑2–style autoregressive language model.
- Train the model using a standard training loop with the AdamW optimizer.
- Generate text using greedy decoding.

## Project Structure

- **Data Preparation:**  
  Downloads and processes the Tiny Shakespeare dataset, tokenizing text at the character level.

- **Model Components:**  
  - `MultiHeadSelfAttention`: Implements scaled dot-product attention with multiple heads.
  - `TransformerDecoderBlock`: A single decoder block that applies masked self-attention, feed-forward layers, dropout, and layer normalization.
  - `GPT2`: Stacks multiple custom decoder blocks, adds positional embeddings, and outputs token logits.

- **Training & Generation:**  
  The training loop uses cross-entropy loss and the AdamW optimizer. A text generation function generates output by greedily selecting the most probable next token.

## Requirements

- Python 3.6+
- [PyTorch](https://pytorch.org/) (1.7 or later recommended)
- [Requests](https://pypi.org/project/requests/)
- [NumPy](https://numpy.org/)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/gpt2-transformer-decoder.git
   cd gpt2-transformer-decoder
