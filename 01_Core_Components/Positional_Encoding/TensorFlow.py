import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model

        # Compute the positional encodings once
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        
        angle_rads = pos * angle_rates

        # Apply sin to even indices; cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        self.pos_encoding = pos_encoding[tf.newaxis, ...]

    def call(self, x):
        # x shape: [batch, seq_len, d_model]
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]