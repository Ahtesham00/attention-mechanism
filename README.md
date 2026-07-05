# Attention Mechanism

Welcome to the **Attention Mechanism** repository. This project is a from-scratch, ground-up implementation of the attention mechanisms that power modern Transformers — every notebook here is built by hand, one step at a time, rather than assembled from a pre-built library.

Attention is the foundational building block for modern deep learning architectures, most notably Transformers (such as BERT, GPT, etc.). This repository contains a series of Jupyter Notebooks that derive and implement each attention variant from first principles using PyTorch, starting from the absolute basics and building up toward the mechanisms used in today's state-of-the-art LLMs.

## Why this Repository Exists

Understanding attention can be daunting because most explanations either wave their hands at the math or drop you straight into a fully assembled Transformer implementation. This repository takes the opposite approach: every mechanism is derived and coded from the ground up, one concept at a time. Every claim is backed by running code — nothing is asserted without being verified numerically first.

## What You Can Expect From Every Notebook

Each notebook in this repository follows the same rigorous, hands-on structure:
- **Plain-language "why," not just "what":** every mathematical step (dot products, scaling, softmax, masking, reshaping) is explained in intuitive, jargon-free terms, aimed at a reader seeing the concept for the first time.
- **Built with `for` loops first:** every computation is first implemented with explicit, element-by-element loops, so you can see exactly which numbers are being multiplied and added.
- **Then rewritten and verified:** each loop-based implementation is rewritten using fast, vectorized PyTorch operations, and cross-checked against the loop version with `torch.allclose` / `torch.equal` so you can trust the shortcut is doing exactly what the loop did.
- **Packaged into a reusable module:** once verified by hand, each mechanism is wrapped into a clean, reusable `nn.Module` class, in the same style used in real model code.

## What's Inside

The repository is structured to take you on a journey through different variations of the attention mechanism, each one built directly on top of the last.

### 1. Vanilla Dot-Product Attention (Complete)
**Notebook:** [`vanilla-dot-product-attention.ipynb`](vanilla-dot-product-attention.ipynb)

The most basic form of self-attention, implemented **without any learnable parameters**:
- **Token Embeddings:** Representing words as numerical vectors.
- **Calculating Attention Scores:** Using dot products (both via loops and matrix multiplication) to determine the relationships between words.
- **Normalizing Attention Scores:** Applying the Softmax function to turn raw scores into attention weights.
- **Calculating Context Vectors:** Creating context-aware representations by taking a weighted sum of the original embeddings.

### 2. Self-Attention with Trainable Weights (Complete)
**Notebook:** [`self-attention-with-trainable-weights.ipynb`](self-attention-with-trainable-weights.ipynb)

Extends vanilla attention with the three learnable projection matrices used in real attention layers:
- **Query, Key, and Value Projections:** Deriving $W_q$, $W_k$, and $W_v$ and explaining, in plain terms, why a model needs three separate "lenses" instead of one.
- **Scaled Dot-Product Attention:** Computing attention scores from queries and keys, then scaling by $\sqrt{d_k}$ — with a from-scratch numerical demonstration of *why* the scaling factor is needed to keep softmax and training well-behaved.
- **A Reusable `SelfAttention` Module:** Packaging the mechanism first with raw `nn.Parameter` matrices, then with `nn.Linear`, and proving both give identical results once their weights are aligned.

### 3. Causal (Masked) Attention (Complete)
**Notebook:** [`causal-attention.ipynb`](causal-attention.ipynb)

Restricts each token to only attend to itself and the tokens before it — the mechanism that makes autoregressive, GPT-style generation possible:
- **Why Hide the Future:** An intuitive explanation of why training must match how the model is actually used at generation time.
- **Two Masking Strategies:** Building the lower-triangular mask by hand, masking-then-renormalizing, and the more efficient $-\infty$-before-softmax trick — with a worked proof that both approaches produce identical results.
- **Dropout:** Explaining and implementing the regularization technique used to prevent the model from over-relying on any single connection.
- **A Reusable `CausalAttention` Module:** Supporting full batches of input, with a `register_buffer`-based mask.

### 4. Multi-Head Attention (Complete)
**Notebook:** [`multi-head-attention.ipynb`](multi-head-attention.ipynb)

Runs several attention computations in parallel so the model can capture different kinds of relationships at once:
- **The Wrapper Approach:** Building multi-head attention the intuitive way, by stacking independent `CausalAttention` instances.
- **The Efficient, Split-Weights Approach:** Deriving — and numerically proving — why one large matrix multiplication followed by a reshape produces identical results to several smaller ones, then building the full `MultiHeadAttention` class around that insight.
- **Every Reshape Explained:** `.view()`, `.transpose()`, and `.contiguous()` are each demonstrated and justified with small, standalone examples before being used in the final class.
- **End-to-End Proof:** A from-scratch, per-head recomputation that confirms the efficient implementation computes exactly what independent single-head attention modules would.

## Roadmap — Coming Next

The next phase of this repository moves from foundational attention into the variants used in modern, production-scale LLMs:

- [ ] Multi-Query Attention
- [ ] Grouped-Query Attention (GQA) — The New Standard
- [ ] Flash Attention
- [ ] Sliding Window Attention (SWA)
- [ ] Multi-Head Latent Attention (MLA) — DeepSeek's Innovation
- [ ] Hybrid / Linear Attention (Gated DeltaNet + Full Attention)
- [ ] DeepSeek Sparse Attention (DSA) — Cutting Long-Context Costs

Each of these will follow the same approach as the notebooks above: derived from first principles, implemented and verified by hand, and only then wrapped into a clean, reusable module.

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
