import tensorflow as tf
from tensorflow.keras import layers


class yogtark:
    def __init__(self):

        model = tf.keras.Sequential()
        layer_1 = layers.Dense(17, shape=(17,), name="input_layer")
        layer_2 = layers.Dense(12, input_shape=(12,), name="hidden1")
        layer_3 = layers.Dense(12, input_shape=(12,))
        layer_4 = layers.Dense(4, input_shape=(12,))

        model.add(layer_1)
        # model.add(layer_2)
        # model.add(layer_3)
        # model.add(layer_4)

        model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.optimizers.Adam(),)

        self.model = model

    def print(self):
        tf.print(self.model)
        print('Hi I am a smart classfier')

    def getresults(self, x):
        return self.model(x)
