import copy
from enum import IntEnum
import numpy as np

from .Model import Model


class Compartment(IntEnum):
    SUSCEPTIBLE = 0
    INFECTIOUS = 1
    RECOVERED = 2


class SIR(Model):
    def __init__(self, h, beta, gamma):
        self.h = h
        self.beta = beta
        self.gamma = gamma
        self.exphgamma = np.exp(-h * gamma)
        self.compExphgamma = 1 - self.exphgamma

    def step(self, curState):
        varlambda = self.beta * curState[Compartment.INFECTIOUS]
        pStar = 1 - np.exp(-self.h * varlambda)
        Inew = np.random.binomial(curState[Compartment.SUSCEPTIBLE],
                                  pStar)
        Rnew = np.random.binomial(curState[Compartment.INFECTIOUS],
                                  self.compExphgamma)

        # Compartmental updating rules
        newState = copy.copy(curState)
        newState[Compartment.SUSCEPTIBLE] -= Inew
        newState[Compartment.INFECTIOUS] += Inew - Rnew
        newState[Compartment.RECOVERED] += Rnew
        return newState
