import tensorflow as tf

class DecoderBlock(tf.keras.layers.Layer):
    """
    A single Decoder Layer in TensorFlow.
    Consists of: 
    1. Masked Self-Attention (Target sequence)
    2. Cross-Attention (Decoder-Encoder bridge)
    3. Position-wise Feed Forward Network
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderBlock, self).__init__()

        # Multi-Head Attention layers
        # Ensure your custom MultiHeadAttention class is accessible
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        # Feed Forward Network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'), # Intermediate layer
            tf.keras.layers.Dense(d_model)                 # Output layer
        ])

        # Normalization and Dropout
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Args:
            x: Target sequence input [batch, target_seq_len, d_model]
            enc_output: Output from the Encoder [batch, input_seq_len, d_model]
            training: Boolean, whether the model is in training mode
            look_ahead_mask: Mask to hide future tokens
            padding_mask: Mask to hide encoder padding
        """
        
        # 1. Masked Self-Attention
        # attn1 shape: [batch, target_seq_len, d_model]
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # 2. Cross-Attention
        # Query: out1 (from Decoder), Key/Value: enc_output (from Encoder)
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # 3. Position-wise Feed Forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2