from abc import ABC, abstractmethod

@abs.AbstractClass
class BatchSolver(ABC):

    def __init__(self, models_dict, opt_dict):
        self.models_dict = models_dict
        self.opt_dict = opt_dict

    @abstractmethod
    def run(self):
        pass