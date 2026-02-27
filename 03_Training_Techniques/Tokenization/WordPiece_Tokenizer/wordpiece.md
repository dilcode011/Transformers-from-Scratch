# 🧩 WordPiece Tokenization

WordPiece is the subword tokenization algorithm used by **BERT**, **DistilBERT**, and **Electra**. Unlike BPE, which is based on frequency, WordPiece chooses merges that maximize the likelihood of the training data.

## ⚙️ How it Works
1. **Subword Prefixing:** Continuation subwords are prefixed with `##` (e.g., `playing` -> `play`, `##ing`).
2. **Greedy Matching:** During encoding, the algorithm looks for the longest possible substring in the vocabulary from the start of the word.
3. **Special Tokens:** Automatically injects:
    - `[CLS]`: At the beginning of every sequence.
    - `[SEP]`: To mark the end of a sequence or a boundary between two.
    - `[UNK]`: For characters/words not present in the vocabulary.

## 🚀 Framework Integration
This directory provides:
* `WordPiece_Base.py`: The core algorithm logic.
* `WordPiece_Torch.py`: Integration with PyTorch `Dataset`.
* `WordPiece_TF.py`: Integration with TensorFlow `tf.data`.