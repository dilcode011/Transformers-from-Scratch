import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_len = max_len
        
        # Build token to ID map
        self.token_to_id = {token: i for i, token in enumerate(tokenizer.vocab)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.tokenize(text)
        
        # Truncate and map to IDs
        token_ids = [self.token_to_id.get(t, self.token_to_id["[UNK]"]) for t in tokens[:self.max_len]]
        
        # Padding
        padding_len = self.max_len - len(token_ids)
        attn_mask = [1] * len(token_ids) + [0] * padding_len
        token_ids += [self.token_to_id["[PAD]"]] * padding_len
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long)
        }