# ✍️ GPT (Generative Pre-trained Transformer)

**Type:** Decoder-Only (Autoregressive)  
**Paper:** [Improving Language Understanding by Generative Pre-Training (2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

## 📌 Overview
The GPT series focuses on **Generative AI**. By using a Decoder-only architecture, GPT is designed to predict the next token in a sequence. It is the foundation for modern LLMs like GPT-4 and Claude.

## 🏗️ Architecture Features
* **Causal/Look-Ahead Masking:** GPT uses a triangular mask to ensure that the prediction for a specific word only depends on the words that came before it. It cannot "see" the future.
* **Autoregressive Nature:** The model generates text one token at a time, feeding its own previous output back in as input for the next step.
* **Generative Pre-training:** Trained on massive datasets to perform "Next Token Prediction," which allows it to learn grammar, facts, and reasoning abilities implicitly.

## 🛠️ Implementation Details
This folder contains:
* **`GPT_Core.py`**: Implementation of the GPT stack. Note the absence of the "Cross-Attention" layer found in the Vanilla Transformer, as there is no separate Encoder input.
* **Linear Head:** A language modeling head that projects the hidden state back to the vocabulary size.

## 🚀 Best Use Cases
- Creative Writing & Storytelling
- Code Generation
- Conversational AI (Chatbots)
- Text Summarization