from abc import abstractmethod, ABC


class Model(ABC):
    def safeStep(samePop=True, newState=True):
        pass

    @abstractmethod
    def step(self, currentState):
        return currentState
