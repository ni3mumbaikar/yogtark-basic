import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

t1x = tf.constant([[1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2]
                   ])

tf.print(t1x.shape)

model = Sequential([
    layers.Dense(17, input_shape=(17,), activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(4, activation='softmax')
])

print(model.summary())

t1y = 'test'
