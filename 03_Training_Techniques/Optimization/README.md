# ⚡ Optimization Techniques

Transformers are highly sensitive to hyperparameters. Standard training settings often lead to "divergence" (where the model stops learning or the loss becomes NaN). This directory implements the two essential techniques used to stabilize training.

## 📈 1. Learning Rate Warmup
Standard Optimizers like Adam can be too aggressive in the first few thousand steps. 
- **Warmup:** We start with a learning rate near $0$ and linearly increase it for the first `n` steps.
- **Decay:** After the warmup, we follow an inverse square root decay.
- **Formula:** $lr = d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$

## 🎯 2. Label Smoothing
Standard Cross-Entropy Loss encourages the model to be 100% confident (Logit $\to \infty$). This leads to overfitting.
- **The Fix:** We "smooth" the labels. Instead of a hard target of `1.0`, we use `0.9` and distribute the remaining `0.1` across all other tokens.
- **Result:** The model becomes more adaptable and generalizes better to unseen data.