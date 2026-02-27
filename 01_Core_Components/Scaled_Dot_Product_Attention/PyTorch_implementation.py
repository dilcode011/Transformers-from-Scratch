import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention as described in 'Attention Is All You Need'.
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Shape (batch_size, num_heads, seq_len, d_k)
            key:   Shape (batch_size, num_heads, seq_len, d_k)
            value: Shape (batch_size, num_heads, seq_len, d_v)
            mask:  Optional mask to ignore specific tokens (Shape: batch_size, 1, 1, seq_len)
        Returns:
            output: Weighted sum of values
            attention_weights: The attention scores (useful for visualization)
        """
        d_k = query.size(-1)
        
        # 1. Compute Dot Product scores
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        
        # 2. Scale scores to prevent vanishing gradients
        scores = scores / math.sqrt(d_k)
        
        # 3. Apply Mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax to get probabilities (Attention Weights)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. Multiply by Values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

if __name__ == "__main__":
    # Quick Test Run
    attn = ScaledDotProductAttention()
    
    # Mock Tensors: (Batch=1, Heads=1, Seq_Len=4, Dimension=8)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    
    out, weights = attn(q, k, v)
    
    print("--- Scaled Dot-Product Attention Test ---")
    print(f"Output Shape: {out.shape}")
    print(f"Weights Shape: {weights.shape}")
    print("\nAttention Weights (First Head):\n", weights[0][0])