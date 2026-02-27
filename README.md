# 🚀 Transformers from Scratch
### *Implementing "Attention Is All You Need" in PyTorch & TensorFlow*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Welcome to my dedicated repository for mastering **Transformer Architectures**. This project documents the complete journey of implementing the seminal "Attention Is All You Need" paper from the ground up. By building every component in both **PyTorch** and **TensorFlow**, I explore the mathematical foundations and framework-specific nuances that power modern Large Language Models (LLMs).

---

## 📌 Project Overview
The Transformer architecture revolutionized NLP and Computer Vision. This repository goes beyond simple architecture to cover the entire model lifecycle:
* **Core Mechanics**: Scaled Dot-Product Attention & Multi-Head Projections.
* **Architecture Variants**: Encoder-only (BERT), Decoder-only (GPT), and Full Stack (Vanilla).
* **Data Pipeline**: Custom Tokenizers (BPE & WordPiece) and Special Token handling.
* **Optimization**: Learning Rate Warmup, Label Smoothing, and Cross-Entropy variants.

---

## 🛠️ Repository Structure

```text
Transformers-from-Scratch/
├── 01_Core_Components/          # Atomic Units of Attention
│   ├── PyTorch/                 # MHA & Scaled Dot-Product Implementations
│   ├── TensorFlow/              # Keras-based Core Layers
│   └── Positional_Encoding.py   # Sinusoidal Order Injection logic
├── 02_Architectures/            # Model Assemblies
│   ├── Vanilla_Transformer/     # The 2017 Original (Encoder-Decoder)
│   ├── BERT/                    # Encoder-only (Bidirectional Context)
│   └── GPT/                     # Decoder-only (Autoregressive Generation)
├── 03_Training_Pipeline/        # The "Engine Room"
│   ├── Tokenization/            # BPE & WordPiece implemented from scratch
│   ├── Optimization/            # Warmup Schedulers & Label Smoothing
│   └── Evaluation/              # BLEU Score & Perplexity Metrics
├── 04_Projects/                 # End-to-End Applications
│   ├── Machine_Translation/     # Full Seq2Seq NMT (English-to-Hindi)
│   ├── Sentiment_Analysis/      # BERT-based Text Classification
│   └── GPT_Story_Gen/           # Mini-GPT for Autoregressive Generation
├── 05_Notebooks/                # Visualization & Analysis
│   └── Visualizing_Attention.ipynb
├── README.md
└── requirements.txt
