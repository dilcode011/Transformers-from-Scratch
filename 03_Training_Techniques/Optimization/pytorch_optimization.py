import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# 1. Label Smoothing in PyTorch
# Since version 1.10+, label_smoothing is built into CrossEntropyLoss
def get_loss_criterion(label_smoothing=0.1):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# 2. Warmup Scheduler
class TransformerLR:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        
        # Internal function for the schedule logic
        self.lr_lambda = lambda step: (self.d_model ** -0.5) * \
            min((step + 1) ** -0.5, (step + 1) * (self.warmup_steps ** -1.5))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

    def step(self):
        self.scheduler.step()

# Usage Example:
# optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
# scheduler = TransformerLR(optimizer, d_model=512, warmup_steps=4000)