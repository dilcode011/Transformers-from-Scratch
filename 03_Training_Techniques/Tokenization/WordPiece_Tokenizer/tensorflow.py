import tensorflow as tf

class TFWordPieceProvider:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.token_to_id = {token: i for i, token in enumerate(tokenizer.vocab)}

    def encode_and_mask(self, text):
        # Convert tensor string to python string
        text_str = text.numpy().decode('utf-8')
        tokens = self.tokenizer.tokenize(text_str)
        
        ids = [self.token_to_id.get(t, self.token_to_id["[UNK]"]) for t in tokens[:self.max_len]]
        
        # Create mask
        mask = [1] * len(ids)
        
        # Pad manually for TF py_function compatibility
        padding = [self.token_to_id["[PAD]"]] * (self.max_len - len(ids))
        ids += padding
        mask += [0] * (self.max_len - len(mask))
        
        return tf.constant(ids, dtype=tf.int32), tf.constant(mask, dtype=tf.int32)

def create_bert_dataset(texts, provider, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    
    def tf_map_fn(text):
        input_ids, mask = tf.py_function(provider.encode_and_mask, [text], [tf.int32, tf.int32])
        input_ids.set_shape([provider.max_len])
        mask.set_shape([provider.max_len])
        return {"input_ids": input_ids, "attention_mask": mask}

    return dataset.map(tf_map_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE)