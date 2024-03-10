#!/usr/bin/env python3

from cmsimpy import BECovid19, SIR


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
    print("Simulating")


def simulateSIR():
    model = SIR.SIR(h=1.0 / 24.0,
                    beta=1.0 / 2.0,
                    gamma=0.729)
    trajectories = []
    for i in range(noSims):
        if i % 10 == 0:
            print(f"Simulation no. {i + 1}")
        state = {SIR.Compartment.SUSCEPTIBLE: 10,
                 SIR.Compartment.INFECTIOUS: 100,
                 SIR.Compartment.RECOVERED: 0}
        trajectories.append([state])
        while state[SIR.Compartment.INFECTIOUS] > 0:
            state = model.step(state)
            trajectories[-1].append(state)

    # Now we can plot the averages per timestep


if __name__ == "__main__":
    simulateSIR()
    exit(0)
