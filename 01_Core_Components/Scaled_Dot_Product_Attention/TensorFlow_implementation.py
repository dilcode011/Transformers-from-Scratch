import tensorflow as tf
import numpy as np

class ScaledDotProductAttention(tf.keras.layers.Layer):
    """
    Computes Scaled Dot-Product Attention in TensorFlow.
    As described in 'Attention Is All You Need'.
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def call(self, query, key, value, mask=None):
        """
        Args:
            query: Shape (batch_size, num_heads, seq_len, d_k)
            key:   Shape (batch_size, num_heads, seq_len, d_k)
            value: Shape (batch_size, num_heads, seq_len, d_v)
            mask:  Float tensor with shape broadcastable 
                   to (..., seq_len, seq_len). Defaults to None.
        Returns:
            output: Weighted sum of values
            attention_weights: The attention scores
        """
        # 1. Compute Dot Product scores
        # query shape: (..., seq_len_q, d_k), key shape: (..., seq_len_k, d_k)
        # transpose_b=True performs the matrix multiplication QK^T
        matmul_qk = tf.matmul(query, key, transpose_b=True)

        # 2. Scale scores by sqrt(d_k)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 3. Apply Mask (if provided)
        # In TF, we usually add a large negative number to masked positions
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # 4. Softmax to get probabilities (Attention Weights)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # 5. Multiply by Values
        output = tf.matmul(attention_weights, value)

        return output, attention_weights

if __name__ == "__main__":
    # Quick Test Run
    attn_layer = ScaledDotProductAttention()

    # Mock Tensors: (Batch=1, Heads=1, Seq_Len=4, Dimension=8)
    q = tf.random.normal((1, 1, 4, 8))
    k = tf.random.normal((1, 1, 4, 8))
    v = tf.random.normal((1, 1, 4, 8))

    out, weights = attn_layer(q, k, v)

    print("--- TensorFlow Scaled Dot-Product Attention Test ---")
    print(f"Output Shape: {out.shape}")
    print(f"Weights Shape: {weights.shape}")
    print("\nAttention Weights (First Head):\n", weights[0][0].numpy())