import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        # Import your previous MHA class here
        # self.mha = MultiHeadAttention(d_model, num_heads) 
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. Self-Attention + Residual + Norm
        # Note: In a real repo, ensure MultiHeadAttention is imported
        attn_output = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed Forward + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x