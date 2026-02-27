import tensorflow as tf

class BertClassifier_TF(tf.keras.Model):
    def __init__(self, vocab_size, num_classes=2, d_model=512, n_layers=6, n_heads=8, dff=2048, max_len=512):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_emb = tf.keras.layers.Embedding(max_len, d_model)
        
        self.encoder_layers = [EncoderBlock(d_model, n_heads, dff) for _ in range(n_layers)]
        
        self.pooler = tf.keras.layers.Dense(d_model, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        
        x = self.token_emb(x) + self.pos_emb(positions)
        
        for layer in self.encoder_layers:
            x = layer(x, training, mask)
            
        # CLS token is at index 0
        cls_token = x[:, 0, :]
        
        pooled = self.pooler(cls_token)
        pooled = self.dropout(pooled, training=training)
        return self.classifier(pooled)