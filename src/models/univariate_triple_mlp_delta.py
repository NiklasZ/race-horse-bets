import tensorflow as tf


class UnivariateTripleMLPDelta(tf.keras.Model):
    # Parameters based on tuner results from 2021-09-02T15-15-45
    def __init__(self, dense1_units=132, dense2_units=20, activation='relu'):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=dense1_units, activation=activation)
        self.dense2 = tf.keras.layers.Dense(units=dense2_units, activation=activation)
        self.dense3 = tf.keras.layers.Dense(units=1, activation='linear')
        self.tuned_learning_rate = 1e-3

    def call(self, inputs):
        """
        Trains a 3 layers of weights corresponding to the input. Corresponds
        to a weighted sum of the change between the inputs.
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """

        def operation(bet_cycles):
            # TODO implement as a layer
            first_elements = tf.identity(bet_cycles)[:-1]
            second_elements = tf.identity(bet_cycles)[1:]
            difference = second_elements - first_elements
            o1 = self.dense1(tf.expand_dims(difference, axis=0))
            o2 = self.dense2(o1)
            o3 = self.dense3(o2)
            return tf.squeeze(o3) + bet_cycles[-1]

        result = tf.map_fn(operation, tf.transpose(inputs[:, :]))

        output = tf.expand_dims(tf.convert_to_tensor(result), axis=0)
        return output


def compile_model(hp):
    hp_dense1_units = hp.Int('units_d1', min_value=4, max_value=312, step=16)
    hp_dense2_units = hp.Int('units_d2', min_value=4, max_value=312, step=16)
    hp_activation_function = hp.Choice('activation_function', values=['relu', 'sigmoid', 'swish'])

    model = UnivariateTripleMLPDelta(hp_dense1_units, hp_dense2_units, hp_activation_function)
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
