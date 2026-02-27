# 🌍 Project NMT: Neural Machine Translation

This project implements the complete **Vanilla Transformer** for translating text between languages. It is the full realization of the "Attention Is All You Need" paper.

## 🏗️ Project Architecture
- **Model:** Full Encoder-Decoder Stack (6 Layers each).
- **Mechanism:** The Encoder creates a "thought vector" (context), and the Decoder generates the translation using **Cross-Attention**.
- **Dataset:** (Example) Multi30k (English-German) or IIT Bombay English-Hindi Corpus.

## 🚀 Key Features
1. **Source-Target Bridge:** Using Cross-Attention to align words in different languages.
2. **Greedy & Beam Search:** Implementing advanced decoding algorithms to find the most likely translation.
3. **Teacher Forcing:** Using the ground-truth target tokens during training to speed up convergence.

## 📂 Folder Structure
- `transformer_nmt.py`: The assembly of Encoder and Decoder.
- `translator_utils.py`: Beam search logic and BLEU score integration.
- `train_nmt.py`: The translation training loop.