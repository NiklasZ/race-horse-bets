import tensorflow as tf


class UnivariateSingleMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=1, activation='linear')

    def call(self, inputs):
        """
        Trains a single layer of weights corresponding to the input. Corresponds
        to a weighted sum of the inputs. Is applied once per bet pool, rather than all at once.
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """

        def operation(bet_cycles):
            return tf.squeeze(self.dense1(tf.expand_dims(bet_cycles, axis=0)))

        result = tf.map_fn(operation, tf.transpose(inputs[:, :]))

        output = tf.expand_dims(tf.convert_to_tensor(result), axis=0)
        return output
