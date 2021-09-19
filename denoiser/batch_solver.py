from abc import ABC, abstractmethod

class BatchSolver(ABC):

    @abstractmethod
    def __init__(self, args):
        pass

    @property
    @abstractmethod
    def models_dict(self):
        pass

    @property
    @abstractmethod
    def opt_dict(self):
        pass

    @abstractmethod
    def run(self, data):
        pass

    @abstractmethod
    def get_eval_loss(self, losses_dict):
        pass