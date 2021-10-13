from abc import ABC, abstractmethod

from denoiser.utils import serialize_model, copy_state

class BatchSolver(ABC):
    @abstractmethod
    def __init__(self, args):
        self.args = args
        self.valid_length = 0

    def train(self):
        for model in self.get_models().values():
            model.train()

    def eval(self):
        for model in self.get_models().values():
            model.eval()

    def serialize(self):
        serialized_models = {}
        serialized_optimizers = {}
        for name, model in self.get_models().items():
            serialized_models[name] = serialize_model(model)
        for name, optimizer in self.get_optimizers().items():
            serialized_optimizers[name] = optimizer.state_dict()
        return serialized_models, serialized_optimizers

    def load(self, package, load_best=False):
        if load_best:
            for name, model_package in package['best_states']['models'].items():
                self.get_models()[name].load_state_dict(model_package['state'])
        else:
            for name, model_package in package['models'].items():
                self.get_models()[name].load_state_dict(model_package['state'])
            for name, opt_package in package['optimizers'].items():
                self.get_optimizers()[name].load_state_dict(opt_package)

    def copy_models_states(self):
        states = {}
        for name, model in self.get_models().items():
            states[name] = copy_state(model.state_dict())
        return states

    @abstractmethod
    def get_models(self) -> dict:
        pass

    @abstractmethod
    def get_optimizers(self) -> dict:
        pass

    @abstractmethod
    def get_losses_names(self) -> list:
        pass

    @abstractmethod
    def set_target_training_length(self, target_length):
        pass

    @abstractmethod
    def calculate_valid_length(self, length):
        pass

    @abstractmethod
    def run(self, data, cross_valid=False):
        pass

    @abstractmethod
    def get_evaluation_loss(self, losses_dict):
        pass

    @abstractmethod
    def get_generator_for_evaluation(self, best_states):
        """
        loads the best state dict seen so far and returns a generator model ready for evaluation
        """
        pass