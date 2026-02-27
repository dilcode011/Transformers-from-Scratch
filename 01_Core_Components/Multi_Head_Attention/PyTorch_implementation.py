import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V and Output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear Projections
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        
        # 2. Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # 3. Scaled Dot-Product Attention
        values, attn = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 4. Concatenate heads
        concat = values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. Final Linear layer
        return self.W_o(concat)

# Quick Test
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(1, 10, 512) # (Batch, Seq, Dim)
print("PyTorch MHA Output Shape:", mha(x, x, x).shape)