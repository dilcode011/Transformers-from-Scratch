# 📊 Evaluation Metrics

How do we know if our Transformer is actually learning? We use two primary metrics to judge performance.

## 1. Perplexity (PPL)
Perplexity is the inverse probability of the test set, normalized by the number of words. 
- **The Intuition:** If a model has a perplexity of 10, it is as if the model was choosing between 10 different words at each step.
- **Lower is better:** A lower PPL means the model is less "perplexed" (more confident).

## 2. BLEU Score (Bilingual Evaluation Understudy)
BLEU is the gold standard for Machine Translation and text summarization.
- **How it works:** It counts how many $n$-grams (1-word, 2-word, 3-word, and 4-word sequences) in the model's output appear in the human reference translation.
- **Brevity Penalty:** It penalizes the model if it generates very short sentences just to get a high precision score.
- **Scale:** 0 to 1 (or 0 to 100). Higher is better.