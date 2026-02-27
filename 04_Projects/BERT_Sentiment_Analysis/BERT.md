# 🎭 Project BERT: Sentiment Analysis

This project implements a **BERT-style Encoder** from scratch to perform binary text classification. We transition from predicting tokens to predicting labels (Positive/Negative).

## 🏗️ Project Architecture
- **Model:** 6-Layer Encoder-only Transformer.
- **Pooling Layer:** We extract the hidden state of the `[CLS]` token, which acts as a summary of the entire sentence.
- **Classification Head:** A Fully Connected layer that maps the `[CLS]` representation to class probabilities.
- **Training Data:** (Example) IMDb Movie Reviews or Twitter Sentiment dataset.

## 🚀 Key Features
1. **Bidirectional Context:** Unlike GPT, BERT looks at the whole sentence at once to understand nuance (e.g., "not bad" vs "bad").
2. **Special Token Logic:** Utilizing the `[CLS]` token for sequence-level classification.
3. **Fine-tuning Pipeline:** Demonstrating how a pre-trained encoder can be adapted for specific tasks.

## 📂 Folder Structure
- `model.py`: BERT architecture with the classification head.
- `train_classifier.py`: Training loop for supervised learning.
- `inference.py`: Script to test the model on custom sentences.