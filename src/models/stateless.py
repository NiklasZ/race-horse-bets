# Models that don't keep any memory of previous models (there is no training)
# Instead, they make a trend prediction only on the input.
import tensorflow as tf


# 1-step linear prediction
class LinearSingleStep(tf.keras.Model):
    def __init__(self):
        """
        Naive model that assumes the same bet change from the previous 2 points
        """
        super().__init__()

    def call(self, inputs):
        """
        Called to output a prediction
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """
        print(inputs)
        previous_states = inputs[-2:, :]
        change = previous_states[1] - previous_states[0]
        result = previous_states[1] + change

        return tf.expand_dims(result, axis=0)
