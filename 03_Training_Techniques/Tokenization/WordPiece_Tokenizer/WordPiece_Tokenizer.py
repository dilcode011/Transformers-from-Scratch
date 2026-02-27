import collections

class WordPieceTokenizer:
    def __init__(self, vocab=None):
        # Initializing with core special tokens
        self.vocab = vocab or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.unk_token = "[UNK]"

    def _tokenize_word(self, word):
        """Greedy matching for subwords."""
        chars = list(word)
        if not chars:
            return []
        
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                return [self.unk_token]
            
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens

    def tokenize(self, text):
        """Main entry point: adds [CLS] and [SEP] logic."""
        words = text.split()
        output_tokens = ["[CLS]"]
        for word in words:
            output_tokens.extend(self._tokenize_word(word))
        output_tokens.append("[SEP]")
        return output_tokens

# --- Quick Test ---
if __name__ == "__main__":
    # Mocking a small BERT-like vocab
    sample_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "learning", "deep", "##ning", "is", "fun"]
    tokenizer = WordPieceTokenizer(vocab=sample_vocab)
    
    text = "learning deep is fun"
    tokens = tokenizer.tokenize(text)
    print(f"Original Text: {text}")
    print(f"WordPiece Tokens: {tokens}")