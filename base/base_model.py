from collections import OrderedDict

import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.modules = list()  # list of module names

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def __setattr__(self, name, value):
        """Set an attribute as in its parent class `nn.Module` and build the
        hierarchy at the same time"""
        # Set module as in parent class `nn.Module`
        super(BaseModel, self).__setattr__(name, value)
        # Build the hierarchy
        if isinstance(value, nn.Module):
            self.modules.append(name)

    def __delattr__(self, name):
        """If the attribute to delete is a layer, delete it from cache as well
        """
        if name in self.modules:
            self.modules.remove(name)
        super(BaseModel, self).__delattr__(name)

    def __getitem__(self, key):
        """Get item as if this is a traditional dictionary, except that key
        could be an integer or a string.

        Parameters
        ----------
        key : int or str
            If int, take it as the index.
            If str, take it as a key as usual.

        """
        if isinstance(key, int):
            key = self.modules[key]
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def _get_layer_name(self, layer):
        """Helper function to get layer name"""
        if isinstance(layer, int):
            layer = self.modules[layer]
        return layer

    def forward(self, inp, output_layer=-1):
        """Forward pass logic. This will pass sequentially in the order in
        which the modules have been set. Pass `output_layer` with layer name or
        layer index to get the output at a specific layer.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor.
        output_layer : int, str, list
            If `int`, passed as layer index.
            If `str`, passed as layer name.
            If `list` of `int` or `str`, passed respectively and return a tuple
            of respective output layers.
            (default: -1 (i.e., the last layer))

        Returns
        -------
            Model output.

        """
        is_list = True
        if isinstance(output_layer, (int, str)):
            is_list = False
            output_layer = [output_layer]
        # Get all output layer names
        output_layer = [self._get_layer_name(layer) for layer in output_layer]
        # Check validity
        for layer in output_layer:
            if layer not in self.modules:
                raise ValueError("Invalid layer name. Expected one of {}, got "
                                 "'{}' instead.".format(self.modules, layer))
        # Forward
        results = [0] * len(output_layer)
        out = inp
        for module in self.modules:
            out = self.__getattr__(module)(out)
            if module in output_layer:
                # We want `results` to be in the same order as `output_layer`
                for i, layer in enumerate(output_layer):
                    if layer == module:
                        results[i] = out
        # Post-process results
        if is_list:
            return tuple(results)
        else:
            return results[0]

    def construct(self, sequence):
        """Automatically construct model based on the OrderedDict passed.

        Parameters
        ----------
        sequence : OrderedDict
            Ordered dictionary of (module_name, module).

        """
        for module_name, module in sequence.items():
            self.__setattr__(module_name, module)

    def copy_layers(self, model, layers):
        """Copy layers from other model.

        Parameters
        ----------
        model : BaseModel or its subclass
            Model to copy layers from. Must be of type `BaseModel` or its
            subclass.
        layers : list
            List of `int` or `str`.
            If `int`, passed as layer index.
            If `str`, passed as layer name.

        """
        if not isinstance(model, BaseModel):
            raise ValueError("Invalid model type. Expected `BaseModel` or its "
                             "subclass, got `{}` instead.".format(type(model)))
        layers_ = list()
        for layer in layers:
            if isinstance(layer, int):
                layers_.append(model.modules[layer])
            elif isinstance(layer, str):
                layers_.append(layer)
            else:
                raise ValueError(
                    "Invalid layer type. Expected one of [str, int], got {} "
                    "instead.".format(type(layer)))
        for layer in layers_:
            self.__setattr__(layer, model[layer])


class BaseGANComponent(BaseModel):
    """Base class for all discriminators and generators."""
    def __init__(self, optimizer, criterion, weights_init=None):
        """
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
    """Base class for all GAN discriminators."""


class BaseGANGenerator(BaseGANComponent):
    """Base class for all GAN discriminators."""
    def __init__(self, optimizer, criterion, weights_init=None,
                 output_range=(-1.0, 1.0)):
        """
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
            Tuple of (`min_value`, `max_value`) representing the output range
            of the generator. (-1.0, 1.0) by default, which is the output range
            of the Tanh layer.

        Attributes
        ----------
        optimizer
        criterion
        metric
        weights_init
        output_range

        """
        super(BaseGANGenerator, self).__init__(
            optimizer, criterion, weights_init)
        self.output_range = output_range
