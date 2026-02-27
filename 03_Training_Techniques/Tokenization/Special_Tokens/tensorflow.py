import tensorflow as tf

class TFTokenHandler:
    def __init__(self, vocab_map):
        self.pad_id = tf.constant(vocab_map.get("[PAD]", 0), dtype=tf.int32)
        self.cls_id = tf.constant(vocab_map.get("[CLS]", 1), dtype=tf.int32)
        self.sep_id = tf.constant(vocab_map.get("[SEP]", 2), dtype=tf.int32)

    def prepare(self, token_ids, max_len):
        """
        Formats sequence using TensorFlow primitives.
        """
        # Ensure input is a tensor
        ids = tf.convert_to_tensor(token_ids, dtype=tf.int32)
        
        # 1. Truncate
        ids = ids[:max_len - 2]
        
        # 2. Add Special Tokens (Concat)
        cls_t = tf.expand_dims(self.cls_id, 0)
        sep_t = tf.expand_dims(self.sep_id, 0)
        formatted_ids = tf.concat([cls_t, ids, sep_t], axis=0)
        
        # 3. Padding and Masking
        # Using tf.pad is more efficient in TF pipelines
        curr_len = tf.shape(formatted_ids)[0]
        padding_size = max_len - curr_len
        
        padded_ids = tf.pad(formatted_ids, [[0, padding_size]], constant_values=self.pad_id)
        
        # Create mask: 1 where not equal to padding
        mask = tf.cast(tf.math.not_equal(padded_ids, self.pad_id), tf.int32)
        
        return padded_ids, mask