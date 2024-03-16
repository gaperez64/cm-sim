from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers


def createNNP(compartments):
    inputs = keras.Input(shape=(2 * len(compartments),))
    # prepare some factorials, but since there's no
    # factorial function in tensorflow, we compose the log
    # of the gamma function with the exponential (below)
    x = layers.Dense(8, activation="relu")(inputs)
    x = layers.Dense(3, activation=tf.math.lgamma)(x)
    # concatenate all of them, including a copy of the input
    y = layers.Dense(8, activation="relu")(inputs)
    c = layers.Concatenate()([x, y])
    # softmax is a sort of normalized exponential, applied to
    # x it gives us a sort of factorial, applied to y it gives an
    # exponential
    c = layers.Dense(8, activation="relu")(c)
    c = layers.Dense(3, activation=tf.math.exp)(c)
    # we also add a linear function of the inputs
    z = layers.Dense(8, activation="relu")(inputs)
    c = layers.Concatenate()([c, z])
    c = layers.Dense(3, activation="relu")(c)
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
        if curState != newState:
            curState = newState
        else:
            curState = state
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


def pdSuccFreq(model, state, sampleSize, numSucc):
    array = arraySuccFreq(model, state, sampleSize, numSucc)
    sknames = [k.name for k in sorted(state.keys())]
    srcnames = ["src." + s for s in sknames]
    tgtnames = ["tgt." + s for s in sknames]
    colnames = srcnames + tgtnames + ["prob"]
    df = pd.DataFrame(data=array, columns=colnames)
    return df
