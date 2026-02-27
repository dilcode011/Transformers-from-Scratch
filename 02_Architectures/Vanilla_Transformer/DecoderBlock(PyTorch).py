import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    """
    A single Decoder Layer.
    Consists of: 1. Masked Self-Attention, 2. Cross-Attention, 3. Feed Forward
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        
        # Self-attention handles the target sequence
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention connects Decoder to Encoder output
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Position-wise Feed Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, trg_mask):
        """
        Args:
            x: Target embeddings [batch, trg_seq_len, d_model]
            enc_output: Encoder final output [batch, src_seq_len, d_model]
            src_mask: Padding mask for encoder output
            trg_mask: Causal/Look-ahead mask for target sequence
        """
        # 1. Masked Multi-Head Self-Attention
        # We use trg_mask to ensure the model doesn't "see" the future
        _x, _ = self.self_attn(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(_x))
        
        # 2. Multi-Head Cross-Attention
        # Query comes from Decoder (x), Key/Value come from Encoder (enc_output)
        _x, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(_x))
        
        # 3. Feed Forward Network
        _x = self.ffn(x)
        x = self.norm3(x + self.dropout(_x))
        
        return x