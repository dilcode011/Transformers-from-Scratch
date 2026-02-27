import tensorflow as tf

class TransformerNMT_TF(tf.keras.Model):
    def __init__(self, src_vocab, trg_vocab, d_model=512, n_layers=6, n_heads=8, dff=2048, max_len=100):
        super().__init__()
        self.encoder = EncoderStack(n_layers, d_model, n_heads, dff, src_vocab, max_len)
        self.decoder = DecoderStack(n_layers, d_model, n_heads, dff, trg_vocab, max_len)
        self.final_layer = tf.keras.layers.Dense(trg_vocab)

    def call(self, inputs, training):
        # inputs is a tuple: (source, target, src_mask, trg_mask)
        src, trg, src_mask, trg_mask = inputs
        
        enc_output = self.encoder(src, training, src_mask)
        dec_output = self.decoder(trg, enc_output, training, trg_mask, src_mask)
        
        return self.final_layer(dec_output)