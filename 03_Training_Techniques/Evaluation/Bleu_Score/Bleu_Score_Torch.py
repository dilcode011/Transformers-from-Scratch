import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class TorchBleuScore:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.smoother = SmoothingFunction().method1

    def calculate(self, predicted_indices, target_indices):
        """
        Args:
            predicted_indices: Tensor [seq_len]
            target_indices: Tensor [seq_len]
        """
        # Convert IDs back to words, ignoring padding/special tokens
        pred_words = self.tokenizer.decode(predicted_indices.tolist()).split()
        target_words = self.tokenizer.decode(target_indices.tolist()).split()

        # BLEU expects a list of reference sentences
        return sentence_bleu([target_words], pred_words, smoothing_function=self.smoother)