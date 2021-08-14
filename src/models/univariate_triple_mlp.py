import tensorflow as tf


class UnivariateTripleMLP(tf.keras.Model):
    def __init__(self, dense1_units=64, dense2_units=16):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=dense1_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=dense2_units, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=1, activation='linear')

    def call(self, inputs):
        """
        Trains a 3 layers of weights corresponding to the input.
         Is applied once per bet pool, rather than all at once.
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """

        def operation(bet_cycles):
            o1 = self.dense1(tf.expand_dims(bet_cycles, axis=0))
            o2 = self.dense2(o1)
            o3 = self.dense3(o2)
            return tf.squeeze(o3)

        result = tf.map_fn(operation, tf.transpose(inputs[:, :]))

        output = tf.expand_dims(tf.convert_to_tensor(result), axis=0)
        return output


def compile_model(hp):
    hp_dense1_units = hp.Int('units_d1', min_value=4, max_value=256, step=8)
    hp_dense2_units = hp.Int('units_d2', min_value=4, max_value=256, step=8)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model = UnivariateTripleMLP(dense1_units=hp_dense1_units, dense2_units=hp_dense2_units)
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
