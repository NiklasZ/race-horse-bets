import tensorflow as tf


class UnivariateSingleMLPDelta(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=1, activation='linear')
        self.tuned_learning_rate = 1e-3

    def call(self, inputs):
        """
        Trains a single layer of weights corresponding to the input. Corresponds
        to a weighted sum of the change between the inputs.
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """

        def operation(bet_cycles):
            # TODO implement as a layer
            first_elements = tf.identity(bet_cycles)[:-1]
            second_elements = tf.identity(bet_cycles)[1:]
            difference = second_elements - first_elements
            return tf.squeeze(self.dense1(tf.expand_dims(difference, axis=0))) + bet_cycles[-1]

        result = tf.map_fn(operation, tf.transpose(inputs[:, :]))

        output = tf.expand_dims(tf.convert_to_tensor(result), axis=0)
        return output


def compile_model(hp):
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    model = UnivariateSingleMLPDelta()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
