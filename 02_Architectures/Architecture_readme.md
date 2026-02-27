# 🏛️ Transformer Architectures

This directory contains the full assembly of Transformer-based models. We move beyond core components to build complete, functional architectures based on research papers.

## 🏗️ Models Implemented

### 1. [Vanilla Transformer](./Vanilla_Transformer/) 
- **Paper:** *Attention Is All You Need (2017)*
- **Type:** Encoder-Decoder stack.
- **Primary Use:** Neural Machine Translation (NMT).

### 2. [BERT](./BERT/) 
- **Paper:** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)*
- **Type:** Encoder-only.
- **Primary Use:** Text classification, NER, and semantic understanding.

### 3. [GPT](./GPT/) 
- **Paper:** *Improving Language Understanding by Generative Pre-Training (2018)*
- **Type:** Decoder-only.
- **Primary Use:** Text generation and causal modeling.

---

## 🔧 Shared Architecture logic
Every model here follows a similar block structure:
1. **Multi-Head Attention** (Self or Cross)
2. **Add & Norm** (Residual connection + Layer Normalization)
3. **Feed Forward Network** (Two linear layers with ReLU/GELU)

## ⚖️ BERT vs. GPT: Which one to use?

| Feature | BERT | GPT |
| :--- | :--- | :--- |
| **Direction** | Bidirectional (Left + Right) | Unidirectional (Left only) |
| **Objective** | Masked Language Modeling (Fill-in-the-blanks) | Causal Modeling (Next word prediction) |
| **Component** | Encoder Stack | Decoder Stack |
| **Best For** | Classification, NER, Q&A | Creative Writing, Chatbots, Code Generation |