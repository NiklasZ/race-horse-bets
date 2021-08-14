import tensorflow as tf


class UnivariateLinear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=1, activation='linear')

    def call(self, inputs):
        """
        Trains only 1 weight to act as a scalar for all bet pools.
        Basically getting the next point through lazy scaling by some scalar X.
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """

        def operation(p):
            return tf.squeeze(self.dense1(tf.expand_dims(tf.expand_dims(p, axis=0), axis=0)))

        result = tf.map_fn(operation, tf.transpose(inputs[-1:, :]))
        output = tf.expand_dims(tf.convert_to_tensor(result), axis=0)

        return output
