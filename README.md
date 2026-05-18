# Attention Mechanism

Welcome to the **Attention Mechanism** repository. This project exists to provide a clear, step-by-step, and practical understanding of how various attention mechanisms work under the hood.

Attention is the foundational building block for modern deep learning architectures, most notably Transformers (such as BERT, GPT, etc.). This repository contains educational Jupyter Notebooks that walk you through the mathematical and programmatic steps of different attention types using PyTorch, starting from the absolute basics and building up to more complex architectures.

## Why this Repository Exists

Understanding attention can be daunting due to the complex math and large-scale implementations found in full Transformer models. This repository breaks down the mechanisms into their most fundamental components. By isolating each concept into its own notebook, you can focus on learning one piece of the puzzle at a time—from basic dot-product scoring to full multi-head attention.

## What's Inside

The repository is structured to take you on a journey through different variations of the attention mechanism. 

### 1. Vanilla Dot-Product Attention (Available Now)
**Notebook:** [`vanilla-dot-product-attention.ipynb`](vanilla-dot-product-attention.ipynb)

This is the most basic form of self-attention. It takes a simple sequence and demonstrates the core process step-by-step **without any learnable parameters**:
- **Token Embeddings:** Representing words as numerical vectors.
- **Calculating Attention Scores:** Using dot products (both via loops and matrix multiplication) to determine the relationships between words.
- **Normalizing Attention Scores:** Applying the Softmax function to turn raw scores into attention weights.
- **Calculating Context Vectors:** Creating context-aware representations by taking a weighted sum of the original embeddings.

### Upcoming Mechanisms (Work in Progress)
- **Self-Attention with Trainable Weights:** Introducing learnable parameter matrices (Queries, Keys, and Values) to allow the model to learn the optimal way to attend to different parts of the sequence.
- **Causal Attention (Masked Attention):** Implementing sequence masking to prevent the model from "looking ahead" at future tokens, a crucial component for autoregressive models like GPT.
- **Multi-Head Attention:** Splitting the attention mechanism into multiple parallel "heads" to allow the model to capture different types of relationships and nuances simultaneously.

## Getting Started

### Prerequisites

To run the notebooks, you will need:
- Python 3.x
- Jupyter Notebook or JupyterLab
- PyTorch (`torch`)

### Installation & Environment Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/attention-mechanism.git
   cd attention-mechanism
   ```
2. Set up a virtual environment (Recommended):
   This keeps your project dependencies isolated.
   ```bash
   python -m venv venv
   
   # On macOS and Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install torch jupyter
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open the desired notebook and run the cells sequentially to see the attention mechanism in action.

Happy learning!
