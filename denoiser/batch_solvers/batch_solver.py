from abc import ABC, abstractmethod

from denoiser.utils import serialize_model, copy_state

class BatchSolver(ABC):
    @abstractmethod
    def __init__(self, args):
        self.models = {}
        self.optimizers_dict = {}
        self.keys = []

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
        for name, optimizer in self.optimizers_dict.items():
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
                self.optimizers_dict[name].load_state_dict(opt_package)

    def copy_models_states(self):
        states = {}
        for name, model in self.models.items():
            states[name] = copy_state(model.state_dict())
        return states


    @abstractmethod
    def get_losses_names(self):
        pass

    @abstractmethod
    def get_models(self):
        pass

    @abstractmethod
    def get_optimizers(self):
        pass

    @abstractmethod
    def run(self, data, cross_valid=False):
        pass

    @abstractmethod
    def get_eval_loss(self, losses_dict):
        pass