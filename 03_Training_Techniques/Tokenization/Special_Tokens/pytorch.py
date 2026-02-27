import torch

class TorchTokenHandler:
    def __init__(self, vocab_map):
        self.pad_id = vocab_map.get("[PAD]", 0)
        self.cls_id = vocab_map.get("[CLS]", 1)
        self.sep_id = vocab_map.get("[SEP]", 2)

    def prepare(self, token_ids, max_len):
        """
        Formats sequence for BERT-style models.
        """
        # 1. Truncate (leaving room for [CLS] and [SEP])
        ids = token_ids[:max_len - 2]
        
        # 2. Wrap with Special Tokens
        formatted_ids = [self.cls_id] + ids + [self.sep_id]
        
        # 3. Create Attention Mask (1 for content, 0 for padding)
        mask = [1] * len(formatted_ids)
        
        # 4. Pad to max_len
        padding_needed = max_len - len(formatted_ids)
        formatted_ids += [self.pad_id] * padding_needed
        mask += [0] * padding_needed
        
        return {
            "input_ids": torch.tensor(formatted_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long)
        }