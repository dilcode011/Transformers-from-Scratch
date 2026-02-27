import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu

class TFBleuScore:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def calculate(self, pred_tensor, target_tensor):
        """
        Args:
            pred_tensor: Predicted token IDs [batch, seq_len]
            target_tensor: Ground truth token IDs [batch, seq_len]
        """
        # Convert tensors to numpy
        preds = pred_tensor.numpy()
        targets = target_tensor.numpy()
        
        scores = []
        for p, t in zip(preds, targets):
            p_words = self.tokenizer.decode(p).split()
            t_words = self.tokenizer.decode(t).split()
            scores.append(sentence_bleu([t_words], p_words))
            
        return tf.reduce_mean(scores)