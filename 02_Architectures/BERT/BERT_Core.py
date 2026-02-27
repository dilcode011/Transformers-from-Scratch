import torch
import torch.nn as nn

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(BERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # Stack of Encoder Blocks
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Head for Masked Language Modeling
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.mlm_head(x)