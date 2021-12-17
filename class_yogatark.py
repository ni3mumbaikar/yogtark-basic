import tensorflow as tf
from tensorflow.keras import layers
import os
import csv
import numpy as np


class yogtark:
    def __init__(self):
        # New keras sequential model is created
        model = tf.keras.Sequential(name="yogtark")

        layer_1 = layers.Dense(17, name="input_layer", activation='relu')
        layer_3 = layers.Dense(12, name="hidden1", activation='relu')
        layer_4 = layers.Dense(6,  name="hidden2", activation='relu')
        layer_5 = layers.Dense(3, name='output', activation='relu')

        model.add(layer_1)
        model.add(layer_3)
        model.add(layer_4)
        model.add(layer_5)
        # provides other hyperparameters of model
        model.compile(loss='mse',
                      optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
        self.model = model

    def print(self):
        tf.print(self.model)
        print('Hi I am a smart classfier')

    def getresults(self, x):
        return self.model(x)

    # Method to build model with tensorflow fit function
    def train(self):
        training_set = './training_set.csv'
        x = []
        y = []
        # To seperate X and Y that is feature set and label from a csv following code is written
        if os.path.exists(training_set):
            with open(training_set, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                fields = next(csvreader)
                for row in csvreader:
                    row.pop(0)
                    y.append(row.pop())
                    x.append(row)
            # casting lists to numpy array as tf model expects a np array or tensor for x and y
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.int32)
            print(len(x), len(y))

            self.model.fit(x=x, y=y, shuffle=True, epochs=500)

        else:
            print('CSV File not found : Generate CSV first')
