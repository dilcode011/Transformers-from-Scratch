import tensorflow as tf

# 1. Label Smoothing in TensorFlow
def get_tf_loss_object(label_smoothing=0.1):
    return tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, 
        reduction='none',
        # Label smoothing for Sparse requires custom logic or conversion to Categorical
        # Here we use Categorical for native smoothing support
    )
    # Note: If using SparseCategoricalCrossentropy, you'd usually convert 
    # labels to one-hot first to use label_smoothing parameter.

# 2. Warmup Scheduler (Custom Schedule Object)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Usage Example:
# learning_rate = CustomSchedule(d_model=512)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)