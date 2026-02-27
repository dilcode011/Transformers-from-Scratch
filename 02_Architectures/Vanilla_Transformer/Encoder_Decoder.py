import torch
import torch.nn as nn
import math

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        # Note: MultiHeadAttention should be imported from 01_Core_Components
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, trg_mask, src_mask):
        # 1. Masked Self-Attention (Prevents looking at future tokens)
        attn1 = self.masked_attention(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(attn1))
        
        # 2. Cross-Attention (Query from Decoder, Key/Value from Encoder)
        attn2 = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        
        # 3. Feed Forward Network
        ff_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout):
        super(Transformer, self).__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len) # Fixed Sine/Cosine
        
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, src_mask, trg_mask):
        # Encoder Pipeline
        enc_x = self.dropout(self.src_embedding(src) + self.pos_encoding[:, :src.size(1), :])
        for layer in self.encoder_layers:
            enc_x = layer(enc_x, src_mask)
            
        # Decoder Pipeline
        dec_x = self.dropout(self.trg_embedding(trg) + self.pos_encoding[:, :trg.size(1), :])
        for layer in self.decoder_layers:
            dec_x = layer(dec_x, enc_x, trg_mask, src_mask)
            
        return self.fc_out(dec_x)