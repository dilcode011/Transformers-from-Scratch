# 01 Core Components

This directory contains the fundamental building blocks of the Transformer architecture. Before assembling a full model, we must understand how tokens interact and how their positions are preserved.

## 🧠 Components Included:

1. **Scaled Dot-Product Attention**: 
   - The heart of the Transformer. It calculates how much "focus" one token should have on another.
   - Includes scaling by $\sqrt{d_k}$ to prevent gradient vanishing.
   - Supports masking for Decoder-only (Causal) or padding scenarios.

2. **Multi-Head Attention**:
   - Instead of one attention mechanism, we run multiple in parallel.
   - This allows the model to attend to different parts of the sentence (e.g., one head for grammar, one for meaning).

3. **Positional Encoding**:
   - Adds a unique signal to each token based on its position in the sequence using Sine and Cosine functions.