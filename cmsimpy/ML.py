import tensorflow as tf
from keras import layers

def createNNP(compartments):
    inputs = keras.Input(shape=(len(compartments),))
