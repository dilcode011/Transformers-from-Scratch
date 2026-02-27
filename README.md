# 🚀 Transformers from Scratch
### *Implementing "Attention Is All You Need" from the ground up*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Welcome to my dedicated repository for mastering **Transformer Architectures** and **Attention Mechanisms**. This project documents my journey of implementing the seminal "Attention Is All You Need" paper, moving from mathematical foundations to Large Language Models (LLMs).

---

## 📌 Project Overview
The Transformer architecture revolutionized NLP and Computer Vision by replacing recurrence with **Self-Attention**. In this repository, I break down the complexity of:
* **Self-Attention Mechanisms** (The "DNA" of modern AI)
* **Multi-Head Projections** for parallel representation
* **Encoder-Decoder Stacks** for Seq2Seq tasks

---

## 🛠️ Repository Structure

```text
Transformers-from-Scratch/
├── 01_Core_Components/
│   ├── Scaled_Dot_Product_Attention.py   <-- 🏁 Start Here!
│   ├── Multi_Head_Attention.py
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
