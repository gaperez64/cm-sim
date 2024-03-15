#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from cmsimpy import BECovid19, ML, \
                    Sim, SIR


noSims = 1000


def simulateBECovid19():
    model = BECovid19.BECovid19(h=1.0 / 24.0,
                                qcsym=0.051 * 1.52,
                                qcasym=0.051 * 6.06,
                                gamma=0.729,
                                ptheta=0.399,
                                minptheta=0.076,
                                delta1=0.240,
                                delta2=0.743904,
                                psi=0.012096,
                                phiomega=0.07425,
                                minphiomega=0.02475,
                                delta3=0.184075,
                                tau=0.000925,
                                delta4=0.184075)
    initial = {BECovid19.Compartment.SUSCEPTIBLE: 1000,
               BECovid19.Compartment.EXPOSED: 0,
               BECovid19.Compartment.INFD_PRESYM: 0,
               BECovid19.Compartment.INFD_ASYM: 1,
               BECovid19.Compartment.INFD_MILD: 1,
               BECovid19.Compartment.INFD_SEV: 1,
               BECovid19.Compartment.INFD_HOSP: 0,
               BECovid19.Compartment.INFD_ICU: 0,
               BECovid19.Compartment.DEAD: 0,
               BECovid19.Compartment.RECOVERED: 0}

    def endPred(state):
        totInf = [state[BECovid19.Compartment.INFD_PRESYM],
                  state[BECovid19.Compartment.INFD_PRESYM],
                  state[BECovid19.Compartment.INFD_ASYM],
                  state[BECovid19.Compartment.INFD_MILD],
                  state[BECovid19.Compartment.INFD_SEV],
                  state[BECovid19.Compartment.INFD_HOSP],
                  state[BECovid19.Compartment.INFD_ICU]]
        return sum(totInf) == 0

    trajectories = Sim.simulate(model, initial, noSims, endPred)
    Sim.plotTrajectories(trajectories,
                         [k for k in BECovid19.Compartment],
                         model.h)
    print("Simulating")


def initSIR():
    model = SIR.SIR(h=1.0 / 24.0,
                    beta=0.001,
                    gamma=0.2)
    initial = {SIR.Compartment.SUSCEPTIBLE: 1000,
               SIR.Compartment.INFECTIOUS: 2,
               SIR.Compartment.RECOVERED: 0}
    return (model, initial)


def simulateSIR():
    (model, initial) = initSIR()

    def endPred(state):
        return state[SIR.Compartment.INFECTIOUS] == 0

    trajectories = Sim.simulate(model, initial, noSims, endPred)
    Sim.plotTrajectories(trajectories,
                         [k for k in SIR.Compartment],
                         model.h)


def trainSIR():
    (model, initial) = initSIR()
    nn = ML.createNNP(SIR.Compartment)
    nn.summary()
    dataset = ML.pdSuccFreq(model, initial,
                            100, 100)

    # split dataset into training and test data
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # split features from labels
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop("prob")
    test_labels = test_features.pop("prob")
    # print(train_dataset.describe().transpose())

    # test prediction
    print(train_features[1:6])
    print(nn(train_features[1:6]))

    # prepare for training
    nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss="mean_absolute_error"
    )

    history = nn.fit(
        train_features,
        train_labels,
        epochs=100,
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2
    )

    # A bit of the history of training
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    # Visualize losses through training
    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 1])
        plt.xlabel('Epoch')
        plt.ylabel('Error [prob]')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_loss(history)

    # Collect test results
    test_predictions = nn.predict(test_features).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [prob]')
    plt.ylabel('Predictions [prob]')
    lims = [0, 1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [prob]')
    _ = plt.ylabel('Count')
    plt.show()


if __name__ == "__main__":
    trainSIR()
    exit(0)
