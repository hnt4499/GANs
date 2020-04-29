import sys
import six
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseFeatureExtractor:
    """Base class for feature extractors. This class implements some protocols
    allowing sharing activations (features extracted by DL models) between
    different metrics.

    Parameters
    ----------
    model_initializer
        Feature extractor class.
    batch_size : int
        Mini-batch size for feature extractor.
    device : str
        String argument to pass to `torch.device`, for example, "cpu" or
        "cuda".
    init_kwargs
        These keyword arguments will be passed to `model_initializer` to
        initialize feature extraction model.

    Attributes
    ----------
    model
        A PyTorch model for extracting features.
    id_real, id_fake
        An unique ID determining current step of the training process. When
        calling this class instance with a different ID, features will be
        re-computed for the new ID. When calling this class instance with the
        same ID, cached features will be returned.
        This protocol allows sharing features between different metrics by
        using the same ID (for example, current batch index) at a step.
    feats_real : torch.Tensor
        Features extracted from real samples.
    feats_fake : torch.Tensor
        Features extracted from fake samples.
    model_initializer
    init_kwargs
    device

    """
    def __init__(self, model_initializer, batch_size, device="cpu",
                 **init_kwargs):
        self.model = None  # model for extracting features
        self.model_initializer = model_initializer
        self.init_kwargs = init_kwargs

        self.batch_size = batch_size
        self.device = torch.device(device=device)

        self.id_real = None
        self.feats_real = None

        self.id_fake = None
        self.feats_fake = None

    def __call__(self, x, id, real=True):
        """Base function for feed-forwarding input. Overwrite if needed.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        id
            An unique ID with respect to the type of input (real of fake)
            determining current step of the training process.
        real : bool
            Whether the inputs are real samples.

        """
        # Initialize model if needed
        if self.model is None:
            self.model = self.model_initializer(**self.init_kwargs)
            self.model = self.model.to(device=self.device)
        # Real samples
        if real:
            # Compute new batch of inputs
            if self.id_real is None or self.id_real != id:
                self.id_real = id  # update current ID
                self.feats_real = self._forward(x)  # update real features
            return self.feats_real
        # Fake samples
        else:
            # Compute new batch of inputs
            if self.id_fake is None or self.id_fake != id:
                self.id_fake = id  # update current ID
                self.feats_fake = self._forward(x)  # update fake features
            return self.feats_fake

    def _forward(self, x):
        """Helper function to forward batched inputs"""
        num_samples = x.shape[0]
        num_batches = math.ceil(num_samples / self.batch_size)
        feats = list()

        self.model.eval()  # important: don't replace this with torch.no_grad()
        for batch in range(num_batches):
            start = batch * self.batch_size
            end = start + self.batch_size

            feat = self.model(x[start:end].to(device=self.device))
            # If model output is not scalar, apply global spatial average
            # pooling. This happens if you choose too shallow layer.
            shape = list(feat.shape)
            if any(s != 1 for s in shape[2:]):
                feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1))
            # Flatten feature(s)
            feat = feat.cpu().view(shape[0], -1)
            feats.append(feat)

        return torch.cat(feats, dim=0)

    def __eq__(self, other):
        """Base function for comparing this class instance with another object.
        Overwrite if needed.

        Parameters
        ----------
        other
            Python object to compare to.

        """
        # Simply compare all initializer's keyword arguments
        for this_kwarg, this_value in self.init_kwargs.items():
            other_value = other.init_kwargs.get(this_kwarg)
            if other_value != this_value:
                return False
        return True


class BaseFeatureExtractorModule(nn.Module):
    """Base class for feature extractors modules, which wraps any subclass of
    `nn.Module` and implements a convenience protocol to output any
    intermediate layer of the wrapped module, as well as to raise more
    informative exception when a RuntimeError is catched.

    Parameters
    ----------
    output_layer : str
        Name of the layer (one of the children modules) to ouput.
    traverse_level : int
        If None, traverse until reached terminal nodes to collect all child
        layers.
        Otherwise, travese up to depth `traverse_level`.

    Attributes
    ----------
    mods : dict
        A dictionary mapping layer' names with its respective module.
    output_layer
    traverse_level

    """
    def __init__(self, output_layer, traverse_level=None):
        super(BaseFeatureExtractorModule, self).__init__()
        self.mods = OrderedDict()
        self.output_layer = output_layer
        self.traverse_level = traverse_level

    def __setattr__(self, name, value):
        """Set an attribute as in its parent class `nn.module` and build the
        mapping at the same time"""
        # Set module as in parent class `nn.Module`
        super(BaseFeatureExtractorModule, self).__setattr__(name, value)
        # Keep track and build the mapping at the same time
        if isinstance(value, nn.Module):
            if isinstance(self.traverse_level, int):
                # Already 1 level down
                level = self.traverse_level - 1
            else:
                level = None
            self._build_mapping(value, path=[name], level=level)

    def _build_mapping(self, module, path=list(), level=None):
        """Helper function to recursively read current module and build the
        mapping from it"""
        # Reached terminal node
        if (not has_children(module)) or level == 0:
            path = ".".join(path)
            self.mods[path] = module
        # Recursively traverse to its child nodes
        else:
            for child_name, child_module in module.named_children():
                child_level = None if level is None else level - 1
                # Get child's children
                child_path = path + [child_name]
                self._build_mapping(child_module, child_path, child_level)

    def _prune_and_update_mapping(self):
        """Prune unused layers for memory efficiency then update the mapping"""
        # Verify that `output_layer` is a valid layer name
        if self.output_layer not in self.mods:
            raise ValueError(
                "Invalid output layer name. Expected one of {}, got '{}' "
                "instead.".format(list(self.mods.keys()), self.output_layer))
        # Prune unused layers, travese backwards
        for mod_name in list(self.mods.keys())[::-1]:
            if mod_name == self.output_layer:
                break
            path = mod_name.split(".")
            prune(self, path)
        # Free up cache
        torch.cuda.empty_cache()
        # Update mapping
        self.mods = OrderedDict()
        self._build_mapping(self, level=self.traverse_level)

    def _forward(self, inp):
        """Base function that takes an input tensor and return output at the
        "output_layer" layer. Any RuntimeError (e.g., size mismatch) will be
        catched and re-raised with more information.
        Subclass's forward function should only wrap this function, e.g.,
        pre-process inputs."""
        # Verify that `output_layer` is a valid layer name
        if self.output_layer not in self.mods:
            raise ValueError(
                "Invalid output layer name. Expected one of {}, got '{}' "
                "instead.".format(list(self.mods.keys()), self.output_layer))

        output = inp
        last_mod_name = None
        for mod_name, mod in self.mods.items():
            # More informative RuntimeError
            try:
                output = mod(output)
            except RuntimeError:
                # Re-raise exception with more information
                t, v, tb = sys.exc_info()
                msg = ("{}\nError catched at layer '{}'. Last successful "
                       "layer is '{}'".format(v, mod_name, last_mod_name))
                v = RuntimeError(msg)
                six.reraise(t, v, tb)
            last_mod_name = mod_name
            # Stop if reached
            if mod_name == self.output_layer:
                return output


"""
Helper functions to handle sub-modules (layers) of a
`BaseFeatureExtractorModule`
"""


def has_children(module):
    """Helper function to check whether a module has children or not"""
    for child in module.named_children():
        return True
    return False


def prune(module, path):
    """Helper function to prune a sub-module given its path"""
    # Continue traversing
    if len(path) > 1:
        child_name = path[0]
        child_module = getattr(module, child_name)
        prune(child_module, path[1:])
        # Prune next node if it is empty after its children have been removed
        # This is to ensure that pruning does not leave any trace
        if not has_children(child_module):
            delattr(module, child_name)
    # Prune next node
    else:
        delattr(module, path[0])
