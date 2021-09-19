from abc import ABC, abstractmethod

class BatchSolver(ABC):
    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def get_models_dict(self):
        pass

    @abstractmethod
    def get_opt_dict(self):
        pass

    @abstractmethod
    def run(self, data, cross_valid=False):
        pass

    @abstractmethod
    def get_eval_loss(self, losses_dict):
        pass