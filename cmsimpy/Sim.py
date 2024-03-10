import matplotlib.pyplot as plt
from statistics import mean


def simulate(model, initial, noSims, endPred):
    trajectories = []
    for i in range(noSims):
        state = initial
        trajectories.append([state])
        while not endPred(state) > 0:
            state = model.safeStep(state)
            trajectories[-1].append(state)
    return trajectories


def plotTrajectories(trajectories, keys, h):
    # Now we can plot the averages per timestep
    def proj(state, c):
        return state[c]

    maxlen = max([len(t) for t in trajectories])
    pertim = map(lambda i: [t[i] if i < len(t)
                            else t[-1]
                            for t in trajectories],
                 range(maxlen))
    pertim = list(pertim)
    times = list(map(lambda i: i * h, range(maxlen)))
    avergs = []

    for k in keys:
        avergs.append(map(lambda subl:
                          map(lambda x: proj(x, k), subl),
                          pertim))
        avergs[-1] = list(map(mean, avergs[-1]))
        plt.plot(times, avergs[-1], label=k.name)

    plt.xlabel("Time")
    plt.ylabel("Average no. of people")
    plt.legend()
    plt.show()
