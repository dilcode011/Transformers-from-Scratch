# 📍 Positional Encoding

Unlike RNNs or LSTMs, Transformers process all tokens in a sequence simultaneously. This makes them fast, but they lose the sense of **word order**. 

## 🌊 How it works
To solve this, we add a vector to each word embedding that represents its position in the sentence. We use sine and cosine functions of different frequencies:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

### Why Sine and Cosine?
1. **Bounded Values:** The values remain between -1 and 1.
2. **Relative Positions:** The model can easily learn to attend by relative positions because for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.
3. **Long Sequences:** It allows the model to extrapolate to sequence lengths longer than those seen during training.

## 🚀 Usage
Simply add the encoding to your word embeddings before passing them into the Transformer blocks:

```python
# PyTorch Example
embeddings = embedding_layer(input_tokens) # [Batch, Seq, Dim]
pe = get_pe_pytorch(seq_len, d_model)
x = embeddings + pe