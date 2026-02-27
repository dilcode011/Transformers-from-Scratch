class SpecialTokenHandler:
    def __init__(self, pad_id=0, cls_id=1, sep_id=2, mask_id=3):
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.mask_id = mask_id

    def prepare_sequence(self, token_ids, max_len):
        """
        1. Adds [CLS] at start and [SEP] at end
        2. Truncates if too long
        3. Pads with [PAD] if too short
        """
        # Leave room for CLS and SEP
        ids = token_ids[:max_len - 2]
        
        # Build sequence: [CLS] + data + [SEP]
        full_seq = [self.cls_id] + ids + [self.sep_id]
        
        # Mask: 1 for real tokens, 0 for padding
        mask = [1] * len(full_seq)
        
        # Padding
        padding_len = max_len - len(full_seq)
        full_seq += [self.pad_id] * padding_len
        mask += [0] * padding_len
        
        return full_seq, mask

# Test Usage
handler = SpecialTokenHandler()
dummy_ids = [10, 20, 30] # Mock subword IDs
seq, attn_mask = handler.prepare_sequence(dummy_ids, max_len=8)
print(f"Padded Sequence: {seq}")
print(f"Attention Mask: {attn_mask}")