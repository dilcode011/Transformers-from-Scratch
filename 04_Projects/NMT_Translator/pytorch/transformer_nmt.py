import torch
import torch.nn as nn
from core import EncoderBlock, DecoderBlock, PositionalEncoding

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=100):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_stack = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder_stack = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        # 1. Encoder Side
        enc_x = self.src_embedding(src) + self.pos_encoding(src)
        for layer in self.encoder_stack:
            enc_x = layer(enc_x, src_mask)
            
        # 2. Decoder Side
        dec_x = self.trg_embedding(trg) + self.pos_encoding(trg)
        for layer in self.decoder_stack:
            # Note: Cross-attention uses enc_x as Key and Value
            dec_x = layer(dec_x, enc_x, src_mask, trg_mask)
            
        return self.fc_out(dec_x)