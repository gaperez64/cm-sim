#!/usr/bin/env python3

from cmsimpy import BECovid19, Sim, SIR


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


def simulateSIR():
    model = SIR.SIR(h=1.0 / 24.0,
                    beta=0.001,
                    gamma=0.2)
    initial = {SIR.Compartment.SUSCEPTIBLE: 1000,
               SIR.Compartment.INFECTIOUS: 2,
               SIR.Compartment.RECOVERED: 0}

    def endPred(state):
        return state[SIR.Compartment.INFECTIOUS] == 0

    trajectories = Sim.simulate(model, initial, noSims, endPred)
    Sim.plotTrajectories(trajectories,
                         [k for k in SIR.Compartment],
                         model.h)


if __name__ == "__main__":
    simulateBECovid19()
    exit(0)
