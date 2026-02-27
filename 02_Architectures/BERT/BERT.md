# 🕵️ BERT (Bidirectional Encoder Representations from Transformers)

**Type:** Encoder-Only  
**Paper:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805)

## 📌 Overview
BERT changed the NLP landscape by proving that **bidirectional** context is superior for understanding the meaning of text. Unlike previous models that read text left-to-right or right-to-left, BERT reads the entire sequence of words at once.

## 🏗️ Architecture Features
* **Bidirectional Attention:** By using the Encoder stack, BERT allows ogni token to attend to both its left and right neighbors simultaneously.
* **Masked Language Modeling (MLM):** During pre-training, 15% of tokens are "masked" (`[MASK]`), and the model must predict them based solely on context.
* **Next Sentence Prediction (NSP):** BERT is trained to understand relationships between sentences by predicting if Sentence B follows Sentence A.

## 🛠️ Implementation Details
This folder contains:
* **`BERT_Core.py`**: A modular implementation of the BERT stack using the Encoder blocks from our `01_Core_Components`.
* **Pooling Layer:** Implementation of the `[CLS]` token extraction for downstream classification tasks.

## 🚀 Best Use Cases
- Text Classification (Sentiment Analysis)
- Named Entity Recognition (NER)
- Question Answering (SQuAD)
- Semantic Similarity