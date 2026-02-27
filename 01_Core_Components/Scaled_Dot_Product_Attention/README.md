# 🔍 Scaled Dot-Product Attention

The **Scaled Dot-Product Attention** is the fundamental building block of the Transformer architecture. It allows the model to assign importance to different words in a sequence regardless of their distance from each other.

## 📐 The Mathematics
The operation follows the formula:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Key Components:
* **Query ($Q$):** Represents the "focus" word looking for information.
* **Key ($K$):** Represents all words in the sequence being "checked" against the query.
* **Value ($V$):** Represents the actual information content to be extracted.
* **$\sqrt{d_k}$ Scaling:** As the dimension $d_k$ increases, the dot product can grow very large, pushing the softmax function into regions where gradients are extremely small. Scaling by $\sqrt{d_k}$ ensures stable training.

## 🛠️ Implementation Details
This implementation (found in `Scaled_Dot_Product_Attention.py`) supports:
1.  **Batch Processing:** Works with multi-head dimensions.
2.  **Masking:** Supports causal masks (for decoders) or padding masks to ignore `<PAD>` tokens.
3.  **Visualization:** Returns attention weights alongside the output for plotting heatmaps.

## 🚀 How to Use
Import the module into your larger Transformer architecture:

```python
from Scaled_Dot_Product_Attention import ScaledDotProductAttention

attn_layer = ScaledDotProductAttention()
output, weights = attn_layer(Q, K, V, mask=None)