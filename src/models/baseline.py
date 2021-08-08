# Models that don't make any prediction and just use the last available value.
import tensorflow as tf

# 1-step prediction baseline
class NaiveNoChange(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """
        Called to output a prediction
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """
        result = inputs[-1, :]
        return tf.expand_dims(result, axis=0)
