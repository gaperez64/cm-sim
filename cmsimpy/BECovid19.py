import copy
from enum import Enum
import numpy as np

from .Model import Model


# I want to use a dictionary for states, these are the keys
class Compartment(Enum):
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFD_PRESYM = 2
    INFD_ASYM = 3
    INFD_MILD = 4
    INFD_SEV = 5
    INFD_HOSP = 6
    INFD_ICU = 7
    DEAD = 8
    RECOVERED = 9


class BECovid19(Model):
    def __init__(self, popSize, h, qcsym, qcasym, gamma,
                 ptheta, minptheta, delta1, delta2,
                 psi, phiomega, minphiomega, delta3,
                 tau, delta4):
        self.popSize = popSize
        self.h = h
        self.qcsym = qcsym
        self.qcasym = qcasym
        self.gamma = gamma
        self.ptheta = ptheta
        self.minptheta = minptheta
        self.delta1 = delta1
        self.delta2 = delta2
        self.psi = psi
        self.phiomega = phiomega
        self.minphiomega = minphiomega
        self.delta3 = delta3
        self.tau = tau
        self.delta4 = delta4

    def step(self, curState):
        # Binomial sampling
        def sampBin(c, p):
            return np.random.binomial(curState[c],
                                      1 - np.exp(-self.h * p))
        lbda = self.qcasym * (curState[Compartment.INFD_PRESYM] +
                              curState[Compartment.INFD_ASYM]) +\
            self.qcsym * (curState[Compartment.INFD_MILD] +
                          curState[Compartment.INFD_SEV])
        Enew = sampBin(Compartment.SUSCEPTIBLE, lbda)
        Inewpresym = sampBin(Compartment.EXPOSED, self.gamma)
        Inewasym = sampBin(Compartment.INFD_PRESYM, self.ptheta)
        Inewmild = sampBin(Compartment.INFD_PRESYM, self.minptheta)
        Inewsev = sampBin(Compartment.INFD_MILD, self.psi)
        Inewhosp = sampBin(Compartment.INFD_SEV, self.phiomega)
        Inewicu = sampBin(Compartment.INFD_SEV, self.minphiomega)
        Dnewhosp = sampBin(Compartment.INFD_HOSP, self.tau)
        Dnewicu = sampBin(Compartment.INFD_ICU, self.tau)
        Rnewasym = sampBin(Compartment.INFD_ASYM, self.delta1)
        Rnewmild = sampBin(Compartment.INFD_MILD, self.delta2)
        Rnewhosp = sampBin(Compartment.INFD_HOSP, self.delta3)
        Rnewicu = sampBin(Compartment.INFD_ICU, self.delta4)

        # Compartmental updating rules
        newState = copy.copy(curState)
        newState[Compartment.SUSCEPTIBLE] -= Enew
        newState[Compartment.EXPOSED] += Enew - Inewpresym
        newState[Compartment.INFD_PRESYM] += Inewpresym\
            - Inewasym - Inewmild
        newState[Compartment.INFD_ASYM] += Inewasym - Rnewasym
        newState[Compartment.INFD_MILD] += Inewmild\
            - Inewsev - Rnewmild
        newState[Compartment.INFD_SEV] += Inewsev\
            - Inewhosp - Inewicu
        newState[Compartment.INFD_HOSP] += Inewhosp\
            - Dnewhosp - Rnewhosp
        newState[Compartment.INFD_ICU] += Inewicu\
            - Dnewicu - Rnewicu
        newState[Compartment.DEAD] += Dnewhosp + Dnewicu
        newState[Compartment.RECOVERED] += Rnewasym\
            + Rnewmild + Rnewhosp + Rnewicu

        return newState
