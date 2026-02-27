# 🧠 Multi-Head Attention (MHA)

While **Scaled Dot-Product Attention** is the engine, **Multi-Head Attention** is the driver. Instead of performing a single attention function, this module allows the model to jointly attend to information from different representation subspaces at different positions.

---

## 📐 The Mathematics
Multi-Head Attention consists of $h$ parallel "heads." Each head uses a different set of learnable weights to project the input into a unique subspace.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head is calculated as:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Why "Multiple" Heads?
* **Parallel Perspectives:** One head might focus on the relationship between a subject and its verb, while another head focuses on the relationship between adjectives and nouns.
* **Subspace Diversification:** By splitting the $d_{model}$ into $h$ heads, each head has a smaller dimension ($d_k = d_{model} / h$), allowing the model to capture multiple nuances without increasing computational cost.

---

## 🏗️ Architecture Breakdown

1.  **Linear Projections:** The input $Q, K, V$ are passed through three different linear layers.
2.  **Split:** The results are split into $h$ heads to allow parallel attention.
3.  **Scaled Dot-Product Attention:** Each head performs the core attention calculation independently.
4.  **Concatenation:** All heads are merged back into a single vector.
5.  **Final Projection:** A final linear layer ($W^O$) projects the concatenated output back to the original $d_{model}$ size.

---

## 🛠️ Implementation Details
This repository provides implementations for both major deep learning frameworks:

* **[PyTorch Implementation](./PyTorch/Multi_Head_Attention.py):** Uses `nn.Module` and vectorized operations for GPU efficiency.
* **[TensorFlow Implementation](./TensorFlow/Multi_Head_Attention.py):** Uses `tf.keras.layers.Layer` for easy integration into Keras models.

---

## 🚀 Usage

### PyTorch
```python
from Multi_Head_Attention import MultiHeadAttention

# Initialize with d_model=512, num_heads=8
mha = MultiHeadAttention(d_model=512, num_heads=8)
output = mha(query, key, value, mask=None)