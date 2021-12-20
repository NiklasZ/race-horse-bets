import tensorflow as tf


class MultistepDense(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten1 = tf.keras.layers.Flatten(input_shape=(1, 99, 14))
        self.dense2 = tf.keras.layers.Dense(units=112, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=56, activation='relu')
        self.dense4 = tf.keras.layers.Dense(units=14, activation='linear')

    def call(self, inputs):
        """
        A basic 3 layer neural net
        :param inputs: 2D array of [race_cycle, bet_amount]
        :return: [bet_amount]
        """

        # Reshape into [1, race_cycle, bet_amount]
        # Necessary for the flatten operation
        reshaped = tf.expand_dims(inputs, axis=0)
        # Run through layers
        o1 = self.flatten1(reshaped)
        o2 = self.dense2(o1)
        o3 = self.dense3(o2)
        result = self.dense4(o3)
        return result

