from abc import abstractmethod, ABC


class Model(ABC):
    def safeStep(self, curState, samePop=True, newState=True):
        s = curState
        while True:
            oldState = s
            s = self.step(s)
            if samePop and sum(s.values()) != sum(oldState.values()):
                s = oldState
                continue
            if newState and s == oldState:
                continue
            return s

    @abstractmethod
    def step(self, currentState):
        return currentState
