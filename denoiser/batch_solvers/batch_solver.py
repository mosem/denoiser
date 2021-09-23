from abc import ABC, abstractmethod

from denoiser.utils import serialize_model, copy_state

class BatchSolver(ABC):
    @abstractmethod
    def __init__(self, args):
        self.args = args
        self.models = {}
        self.optimizers = {}
        self.losses_names = []
        self.valid_length = 0

    def train(self):
        for model in self.models.values():
            model.train()

    def eval(self):
        for model in self.models.values():
            model.eval()

    def serialize(self):
        serialized_models = {}
        serialized_optimizers = {}
        for name, model in self.models.items():
            serialized_models[name] = serialize_model(model)
        for name, optimizer in self.optimizers.items():
            serialized_optimizers[name] = optimizer.state_dict()
        return serialized_models, serialized_optimizers

    def load(self, package, load_best=False):
        if load_best:
            for name, model_package in package['best_states']['models'].items():
                self.models[name].load_state_dict(model_package['state'])
        else:
            for name, model_package in package['models'].items():
                self.models[name].load_state_dict(model_package['state'])
            for name, opt_package in package['optimizers'].items():
                self.optimizers[name].load_state_dict(opt_package)

    def copy_models_states(self):
        states = {}
        for name, model in self.models.items():
            states[name] = copy_state(model.state_dict())
        return states

    def get_models(self):
        return self.models

    def get_optimizers(self):
        return self.optimizers

    def get_losses_names(self):
        return self.losses_names

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
    def get_generator_model(self):
        pass

    @abstractmethod
    def get_generator_state(self, best_states):
        pass