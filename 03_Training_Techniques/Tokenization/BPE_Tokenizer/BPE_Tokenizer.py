import re
from collections import Counter, defaultdict

class SimpleBPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = []

    def get_stats(self, ids):
        """Counts frequency of adjacent pairs of integers."""
        counts = Counter()
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i+1])] += 1
        return counts

    def merge(self, ids, pair, idx):
        """Replaces all occurrences of 'pair' with the new 'idx'."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        # Initial vocabulary: individual characters (UTF-8 bytes)
        tokens = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256
        ids = list(tokens)

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            # Find the most frequent pair
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = self.merge(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
            print(f"Merge {i+1}: {top_pair} -> {new_id}")

        self.vocab = ids
        return ids

# --- Test ---
if __name__ == "__main__":
    raw_text = "learning deep learning is fun. learning is life."
    tokenizer = SimpleBPETokenizer(vocab_size=265) # Small vocab for demo
    trained_ids = tokenizer.train(raw_text)
    print(f"\nFinal Token IDs: {trained_ids}")