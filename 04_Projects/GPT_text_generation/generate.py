import torch.nn.functional as F

def generate(model, start_ids, max_new_tokens, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        # Predict logits for the last token
        logits = model(start_ids)
        logits = logits[:, -1, :] / temperature
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        # Append to the sequence
        start_ids = torch.cat((start_ids, next_id), dim=1)
        
    return start_ids