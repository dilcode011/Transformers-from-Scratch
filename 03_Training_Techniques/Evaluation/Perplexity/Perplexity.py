import torch
import tensorflow as tf
import numpy as np

# PyTorch Version
def pytorch_perplexity(loss):
    """Calculates perplexity from PyTorch CrossEntropy loss."""
    return torch.exp(loss).item()

# TensorFlow Version
def tensorflow_perplexity(loss):
    """Calculates perplexity from TensorFlow CategoricalCrossentropy loss."""
    return tf.exp(loss).numpy()

# Test logic
if __name__ == "__main__":
    mock_loss = 2.302  # Equivalent to ln(10)
    print(f"PyTorch Perplexity: {np.exp(mock_loss):.2f}")