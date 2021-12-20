import tensorflow as tf


class UnivariateQuintupleMLPDelta(tf.keras.Model):
    def __init__(self, dense1_units=20, dense2_units=20, dense3_units=116, dense4_units=164, activation='relu'):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=dense1_units, activation=activation)
        self.dense2 = tf.keras.layers.Dense(units=dense2_units, activation=activation)
        self.dense3 = tf.keras.layers.Dense(units=dense3_units, activation=activation)
        self.dense4 = tf.keras.layers.Dense(units=dense4_units, activation=activation)
        self.dense5 = tf.keras.layers.Dense(units=1, activation='linear')
        self.tuned_learning_rate = 1e-3

    def call(self, inputs):
        """
        Trains a 5 layers of weights corresponding to the input. Corresponds
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
            o4 = self.dense4(o3)
            o5 = self.dense5(o4)

            return tf.squeeze(o5) + bet_cycles[-1]

        result = tf.map_fn(operation, tf.transpose(inputs[:, :]))

        output = tf.expand_dims(tf.convert_to_tensor(result), axis=0)
        return output


def compile_model(hp):
    hp_dense1_units = hp.Int('units_d1', min_value=4, max_value=312, step=16)
    hp_dense2_units = hp.Int('units_d2', min_value=4, max_value=312, step=16)
    hp_dense3_units = hp.Int('units_d3', min_value=4, max_value=312, step=16)
    hp_dense4_units = hp.Int('units_d4', min_value=4, max_value=312, step=16)
    hp_activation_function = hp.Choice('activation_function', values=['relu', 'sigmoid', 'swish'])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = UnivariateQuintupleMLPDelta(hp_dense1_units, hp_dense2_units, hp_dense3_units, hp_dense4_units,
                                        hp_activation_function)
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
