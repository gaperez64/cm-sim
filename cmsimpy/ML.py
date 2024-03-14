import tensorflow as tf
import keras
from keras import layers


def createNNP(compartments):
    inputs = keras.Input(shape=(2 * len(compartments),))
    # allow for linear transformations of the input
    w = layers.Dense(32, activation="relu")(inputs)
    # prepare some factorials, but since there's no
    # factorial function in tensorflow, we compose the log
    # of the gamma function with the exponential
    y = layers.Dense(16, activation=tf.math.lgamma)(w)
    y = layers.Dense(16, activation=tf.math.exp)(y)
    # prepare some exponentials
    z = layers.Dense(16, activation=tf.math.exp)(w)
    # concatenate all of them
    x = layers.Dense(16, activation="relu")(w)
    c = layers.Concatenate()([x, y, z])
    c = layers.Dense(16, activation="relu")(c)
    # prepare a single output
    outputs = layers.Dense(1)(c)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
