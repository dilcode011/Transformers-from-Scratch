# 🚀 Transformers from Scratch
### *Implementing "Attention Is All You Need" in PyTorch & TensorFlow*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Welcome to my dedicated repository for mastering **Transformer Architectures**. This project documents my journey of implementing the seminal "Attention Is All You Need" paper from the ground up, using both **PyTorch** and **TensorFlow** to understand the cross-framework nuances of Attention Mechanisms.

---

## 📌 Project Overview
The Transformer architecture revolutionized NLP and Computer Vision. In this repository, I break down:
* **Self-Attention Mechanisms** (The "DNA" of modern AI)
* **Multi-Head Projections** for parallel representation
* **Framework Comparisons**: Implementing core logic in both `torch.nn` and `tf.keras`.

---

## 🛠️ Repository Structure

```text
Transformers-from-Scratch/
├── 01_Core_Components/
│   ├── PyTorch/
│   │   ├── Scaled_Dot_Product_Attention.py
│   │   └── Multi_Head_Attention.py
│   ├── TensorFlow/
│   │   ├── Scaled_Dot_Product_Attention.py
│   │   └── Multi_Head_Attention.py
│   └── Positional_Encoding.py
├── 02_Architectures/
│   ├── Vanilla_Transformer/              # The Original Paper (2017)
│   ├── BERT/                             # Encoder-only (Masked LM)
│   └── GPT/                              # Decoder-only (Generative)
├── 03_Projects/
│   ├── Machine_Translation/              # Seq2Seq translation
│   └── Sentiment_Classifier/             # BERT fine-tuning
├── 04_Notebooks/
│   └── Visualizing_Attention_Heads.ipynb
├── README.md
└── requirements.txt
