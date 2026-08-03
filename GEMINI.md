# CS336 Assignment 1: Building a Transformer LM (Project Context)

## Project Overview
This project is part of Stanford's CS336 course (Spring 2025). The goal is to build a standard Transformer language model (LM) from scratch, including its tokenizer, core architecture, optimizer, and training pipeline.

### Key Components to Implement:
1.  **Byte-Pair Encoding (BPE) Tokenizer**: Implementing a byte-level BPE tokenizer and training it on the TinyStories dataset.
2.  **Transformer LM**: Implementing core building blocks (Linear, Embedding, RMSNorm, SwiGLU, Rotary Positional Embeddings - RoPE) and assembling them into a Transformer block and a full Language Model.
3.  **Optimization**: Implementing the Cross-Entropy loss function and the AdamW optimizer from scratch.
4.  **Training & Generation**: Building the training loop (with checkpointing support), data loading, and inference logic (with nucleus sampling/top-p).

## Building and Running

### Environment Management
The project uses `uv` for environment and dependency management.
-   **Run code**: `uv run <python_file_path>`
-   **Install/Sync dependencies**: `uv sync`

### Testing
Unit tests are provided in the `tests/` directory.
-   **Run all tests**: `uv run pytest`
-   **Run specific tests**: `uv run pytest -k <test_name>`
-   **Connecting implementation to tests**: Implementation must be hooked up via `tests/adapters.py`. Each unimplemented component in `adapters.py` raises `NotImplementedError` by default.

### Data
Datasets (TinyStories, OpenWebText) are typically downloaded into a `data/` directory.

## Development Conventions

### Coding Style & Constraints
-   **From-Scratch Ethos**: You are expected to build components from scratch.
-   **Prohibited Modules**: Do NOT use `torch.nn` (except for `Parameter` and containers like `Module`, `ModuleList`, `Sequential`), `torch.nn.functional`, or `torch.optim` (except for the `Optimizer` base class).
-   **Precision**: Use `float32` for most operations. For `RMSNorm`, upcast input to `float32` before normalization and downcast after.
-   **Tensor Operations**: Use `einsum` (via `torch.einsum` or `einops.einsum`) and `einops.rearrange` for ergonomic and self-documenting tensor manipulation, especially when dealing with batch dimensions.
-   **Type Hinting**: Use `jaxtyping` and `numpy.typing` for clear tensor shape and type documentation.

### Architecture Specifics
-   **Linear Layer**: No bias term (consistent with modern LLMs like Llama).
-   **Normalization**: Pre-norm architecture using RMSNorm.
-   **Activation**: SwiGLU (SiLU + Gated Linear Unit).
-   **Position Embeddings**: Rotary Positional Embeddings (RoPE).

## AI Agent Interaction Protocol (via CLAUDE.md)
AI agents (like Gemini) should primarily act as **Teaching Assistants**:
-   **Explain concepts** and guide understanding.
-   **Point to relevant materials** (lecture, handouts, docs).
-   **Review code** and suggest improvements/edge cases without providing direct fixes.
-   **Debug via guiding questions**.
-   **DO NOT** write python/pseudocode for core components, give direct solutions, or edit student code directly unless explicitly requested for specific boilerplate or test-adapter glue code.
