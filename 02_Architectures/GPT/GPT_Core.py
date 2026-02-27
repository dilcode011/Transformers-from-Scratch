import tensorflow as tf

class GPT(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dff, max_len, rate=0.1):
        super(GPT, self).__init__()
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_embedding = tf.keras.layers.Embedding(max_len, d_model)
        
        # Stack of Decoder Blocks (without the Cross-Attention part)
        self.decoder_layers = [DecoderBlock(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.linear_out = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training, look_ahead_mask):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        for layer in self.decoder_layers:
            # Note: GPT only uses self-attention, so enc_output is None
            x, _, _ = layer(x, enc_output=None, training=training, 
                            look_ahead_mask=look_ahead_mask, padding_mask=None)
            
        return self.linear_out(x)