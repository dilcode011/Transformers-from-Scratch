# 🔠 Byte Pair Encoding (BPE) Tokenizer

This directory contains a from-scratch implementation of the **Byte Pair Encoding (BPE)** algorithm, the standard subword tokenization method used by models like GPT-2, GPT-3, and RoBERTa.

## 📘 What is BPE?
BPE solves the "Out-of-Vocabulary" (OOV) problem by breaking words into smaller subword units. It starts with individual characters and iteratively merges the most frequent adjacent pairs.

### The Algorithm:
1. **Initialize:** Every character is a token.
2. **Count:** Find the most frequent adjacent pair of tokens in the corpus.
3. **Merge:** Replace all occurrences of that pair with a new token.
4. **Repeat:** Continue until the desired vocabulary size is reached.

## 🚀 Implementation Files
- `bpe_base.py`: The core algorithm (Training, Encoding, Decoding).
- `pytorch_wrapper.py`: Integration with `torch.utils.data.Dataset`.
- `tensorflow_wrapper.py`: Integration with `tf.data.Dataset` pipelines.