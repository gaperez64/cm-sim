#!/usr/bin/env python3

import matplotlib.pyplot as plt
from statistics import mean

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
                    beta=0.001,
                    gamma=0.2)
    trajectories = []
    for i in range(noSims):
        if i % 10 == 0:
            print(f"Simulation no. {i + 1}")
        state = {SIR.Compartment.SUSCEPTIBLE: 1000,
                 SIR.Compartment.INFECTIOUS: 2,
                 SIR.Compartment.RECOVERED: 0}
        trajectories.append([state])
        while state[SIR.Compartment.INFECTIOUS] > 0:
            state = model.step(state)
            trajectories[-1].append(state)

    # Now we can plot the averages per timestep
    def proj(state, c):
        return state[c]

    maxlen = max([len(t) for t in trajectories])
    pertim = map(lambda i: [t[i] if i < len(t)
                            else t[-1]
                            for t in trajectories],
                 range(maxlen))
    pertim = list(pertim)
    inftim = map(lambda subl:
                 map(lambda x: proj(x, SIR.Compartment.INFECTIOUS), subl),
                 pertim)
    sustim = map(lambda subl:
                 map(lambda x: proj(x, SIR.Compartment.SUSCEPTIBLE), subl),
                 pertim)
    rectim = map(lambda subl:
                 map(lambda x: proj(x, SIR.Compartment.RECOVERED), subl),
                 pertim)
    avginf = map(mean, inftim)
    avgsus = map(mean, sustim)
    avgrec = map(mean, rectim)
    times = list(map(lambda i: i * model.h, range(maxlen)))

    # Plotting
    plt.plot(times, list(avgsus), label="Susceptible")
    plt.plot(times, list(avginf), label="Infectious")
    plt.plot(times, list(avgrec), label="Recovered")
    plt.xlabel("Time")
    plt.ylabel("Average no. of people")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    simulateSIR()
    exit(0)
