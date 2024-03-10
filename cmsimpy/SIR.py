import numpy as np

from .Model import Model


class SIR(Model):
    def __init__(self, popSize, h, beta, gamma):
        self.popSize = popSize
        self.h = h
        self.beta = beta
        self.gamma = gamma
        self.exphgamma = np.exp(-h * gamma)
        self.compExphgamma = 1 - self.exphgamma
        self.knownP = dict()

    def step(self, m):
        (m1, m2, m3) = m
        varlambda = self.beta * m2
        pStar = 1 - np.exp(-self.h * varlambda)
        Inew = np.random.binomial(m1, pStar)
        Rnew = np.random.binomial(m2, self.compExphgamma)

        # Compartmental updating rules
        n1 = m1 - Inew
        n2 = m2 + Inew - Rnew
        n3 = m3 + Rnew
        n = (n1, n2, n3)
        assert sum(n) == self.popSize
        return n
