from collections import defaultdict

import numpy as np
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


def sampleSuccFreq(model, state, sampleSize, numSucc):
    frequencies = {}
    curState = state
    for _ in range(numSucc):
        cnt = 0
        local = defaultdict(int)
        for _ in range(sampleSize):
            newState = model.step(curState)
            local[frozenset(newState.items())] += 1
            if curState != newState:
                frequencies[(frozenset(newState.items()),
                             frozenset(curState.items()))] = 0
            cnt += 1
        for k, v in local.items():
            assert v / cnt <= 1
            frequencies[(frozenset(curState.items()), k)] = v / cnt
        curState = newState
    return frequencies


def arraySuccFreq(model, state, sampleSize, numSucc):
    freqs = sampleSuccFreq(model, state, sampleSize, numSucc)
    data = []
    for k, v in freqs.items():
        (cur, succ) = k
        cur = dict(cur)
        succ = dict(succ)
        sks = sorted(cur.keys())
        row = [cur[o] for o in sks] + [succ[o] for o in sks]
        row.append(v)
        data.append(row)
    return np.array(data)
