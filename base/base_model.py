import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """Forward pass logic.
        Returns
        -------
            Model output.

        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class BaseGANComponent(BaseModel):
    """Base class for all discriminators and generators.

    Parameters
    ----------
    optimizer : fn
        A function initialized in `compile.optimizers` that takes only the
        model trainable parameters as input.
    criterion : fn
        A function initialized in `compile.criterion` that takes the model
        predictions and target labels, and return the computed loss.
    metric : fn
        A function initialized in `compile.metrics` that takes the model
        predictions and target labels, and return the computed metric.
    weights_init : fn
        A function initialized in `models.weights_init` that will then be
        passed to `model.apply()`.

    Attributes
    ----------
    optimizer
    criterion
    metric
    weights_init

    """
    def __init__(self, optimizer, criterion, weights_init=None):
        super(BaseGANComponent, self).__init__()
        # Cache data
        self.optimizer = optimizer
        self.criterion = criterion
        self.weights_init = weights_init

    def init(self):
        """Initialize optimizer and model weights. Called only when model
        definition is done."""
        # Apply optimizer to model trainable parameters
        trainable_params = filter(
            lambda p: p.requires_grad, self.parameters())
        self.optimizer = self.optimizer(trainable_params)
        # Initialize model weights
        if self.weights_init is not None:
            self.apply(self.weights_init)


class BaseGANDiscriminator(BaseGANComponent):
    """Base class for all GAN discriminators.

    Parameters
    ----------
    optimizer : fn
        A function initialized in `compile.optimizers` that takes only the
        model trainable parameters as input.
    criterion : fn
        A function initialized in `compile.criterion` that takes the model
        predictions and target labels, and return the computed loss.
    metric : fn
        A function initialized in `compile.metrics` that takes the model
        predictions and target labels, and return the computed metric.
    weights_init : fn
        A function initialized in `models.weights_init` that will then be
        passed to `model.apply()`.

    Attributes
    ----------
    optimizer
    criterion
    metric
    weights_init

    """


class BaseGANGenerator(BaseGANComponent):
    """Base class for all GAN discriminators.

    Parameters
    ----------
    optimizer : fn
        A function initialized in `compile.optimizers` that takes only the
        model trainable parameters as input.
    criterion : fn
        A function initialized in `compile.criterion` that takes the model
        predictions and target labels, and return the computed loss.
    metric : fn
        A function initialized in `compile.metrics` that takes the model
        predictions and target labels, and return the computed metric.
    weights_init : fn
        A function initialized in `models.weights_init` that will then be
        passed to `model.apply()`.
    output_range : tuple
        Tuple of (`min_value`, `max_value`) representing the output range of
        the generator. (-1.0, 1.0) by default, which is the output range of the
        Tanh layer.

    Attributes
    ----------
    optimizer
    criterion
    metric
    weights_init

    """
    def __init__(self, optimizer, criterion, weights_init=None,
                 output_range=(-1.0, 1.0)):
        super(BaseGANGenerator, self).__init__(
            optimizer, criterion, weights_init)
        self.output_range = output_range
