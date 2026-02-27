import tensorflow as tf

def create_tf_dataset(texts, tokenizer, max_len, batch_size):
    def encode_fn(text):
        # TF requires mapping pure python functions
        tokens = tokenizer.encode(text.decode('utf-8'))
        return tokens[:max_len]

    def py_func_wrapper(text):
        return tf.py_function(encode_fn, [text], tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices(texts)
    dataset = dataset.map(py_func_wrapper)
    # Handle padding via padded_batch
    dataset = dataset.padded_batch(batch_size, padded_shapes=[max_len])
    return dataset

# Usage
# tf_ds = create_tf_dataset(raw_texts, tokenizer, 128, 32)