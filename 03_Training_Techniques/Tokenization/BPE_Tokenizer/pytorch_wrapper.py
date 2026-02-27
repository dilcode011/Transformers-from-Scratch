import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = [self.tokenizer.encode(t) for t in texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx][:self.max_len]
        # Padding logic
        padding = [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens + padding)

# Usage
# dataset = TextDataset(texts, tokenizer, max_len=128)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)