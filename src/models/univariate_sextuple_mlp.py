import tensorflow as tf


class UnivariateSextupleMLP(tf.keras.Model):
    def __init__(self, dense1_units=64, dense2_units=64, dense3_units=32, dense4_units=32, dense5_units=32):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=dense1_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=dense2_units, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=dense3_units, activation='relu')
        self.dense4 = tf.keras.layers.Dense(units=dense4_units, activation='relu')
        self.dense5 = tf.keras.layers.Dense(units=dense5_units, activation='relu')
        self.dense6 = tf.keras.layers.Dense(units=1, activation='linear')

    def call(self, inputs):
        """
        Trains a 6 layers of weights corresponding to the input.
         Is applied once per bet pool, rather than all at once.
         More or less the final attempt at a standard MLP
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """

        def operation(bet_cycles):
            o1 = self.dense1(tf.expand_dims(bet_cycles, axis=0))
            o2 = self.dense2(o1)
            o3 = self.dense3(o2)
            o4 = self.dense4(o3)
            o5 = self.dense5(o4)
            o6 = self.dense6(o5)
            return tf.squeeze(o6)

        result = tf.map_fn(operation, tf.transpose(inputs[:, :]))

        output = tf.expand_dims(tf.convert_to_tensor(result), axis=0)
        return output


def compile_model(hp):
    hp_dense1_units = hp.Int('units_d1', min_value=4, max_value=256, step=8)
    hp_dense2_units = hp.Int('units_d2', min_value=4, max_value=256, step=8)
    hp_dense3_units = hp.Int('units_d3', min_value=4, max_value=256, step=8)
    hp_dense4_units = hp.Int('units_d4', min_value=4, max_value=256, step=8)
    hp_dense5_units = hp.Int('units_d4', min_value=4, max_value=256, step=8)

    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = UnivariateSextupleMLP(dense1_units=hp_dense1_units, dense2_units=hp_dense2_units,
                                dense3_units=hp_dense3_units, dense4_units=hp_dense4_units,
                                dense5_units=hp_dense5_units)
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
