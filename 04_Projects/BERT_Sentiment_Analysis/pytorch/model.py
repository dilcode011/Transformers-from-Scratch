import torch
import torch.nn as nn
# Assumes components from 01_Core_Components
from core import EncoderBlock 

class BertForClassification(nn.Module):
    def __init__(self, vocab_size, num_classes=2, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # BERT is a stack of Encoders
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        # The "Pooler" - takes the [CLS] token (index 0)
        self.pooler = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        
        # The Final Classifier
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        
        for layer in self.encoders:
            x = layer(x, mask)
        
        # Extract [CLS] token: first token of the sequence
        cls_token = x[:, 0, :]
        
        # Pooling & Classification
        pooled_output = self.tanh(self.pooler(cls_token))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits