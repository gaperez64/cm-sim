#!/usr/bin/env python3

from cmsimpy import BECovid19, SIR


def simulateBECovid19():
    model = BECovid19.BECovid19(popSize=10,
                                h=1.0 / 24.0,
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
    model = SIR.SIR(popSize=4,
                    h=1.0/24.0,
                    beta=1.0/2.0,
                    gamma=1.0/6.0)
    print("Simulating")


if __name__ == "__main__":
    simulateSIR()
    exit(0)
