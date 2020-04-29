import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base_fe import BaseFeatureExtractor, BaseFeatureExtractorModule


class InceptionV3FE(BaseFeatureExtractor):
    """InceptionV3 feature extractor.

    Parameters
    ----------
    output_layer : str
        Name of the layer to output.
    resize_to : int
        If None, do not resize input images.
        If not None, bilinearly resizes input images to width and height
        `resize_to` before feeding input to model.
        As the network without fully connected layers is fully
        convolutional, it should be able to handle inputs of arbitrary
        size, so resizing might not be strictly needed. If needed, the
        images should be resized to (299, 299).
    normalize_input : bool
        If true, scales the input from range (0, 1) to the range the
        pretrained Inception network expects, namely (-1, 1).

    batch_size : int
        Mini-batch size for feature extractor.
    device : str
        String argument to pass to `torch.device`, for example, "cpu" or
        "cuda".
    prune : bool
        Whether to free up memory by prune unused layers. (default: False)
    traverse_level : int
        If None, traverse until reached terminal nodes to collect all child
        layers.
        Otherwise, travese up to depth `traverse_level`.

    Attributes
    ----------
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
    model : InceptionV3
        InceptionV3 model.
    batch_size

    """
    def __init__(self, output_layer, resize_to=None, normalize_input=True,
                 batch_size=64, device="cpu", prune=False,
                 traverse_level=None):
        super(InceptionV3FE, self).__init__(
            model_initializer=InceptionV3, batch_size=batch_size,
            device=device, output_layer=output_layer, resize_to=resize_to,
            normalize_input=normalize_input, prune=prune,
            traverse_level=traverse_level
        )

    def __eq__(self, other):
        # Check if `other` is an instance of `InceptionV3FE` or its subclass
        if not isinstance(other, InceptionV3FE):
            return False
        # Simply compare InceptionV3 parameters
        return super(InceptionV3FE, self).__eq__(other)


class InceptionV3(BaseFeatureExtractorModule):
    """Pretrained InceptionV3 for feature extraction."""
    def __init__(self, output_layer, resize_to=None, normalize_input=True,
                 prune=False, traverse_level=None):
        """Build pretrained InceptionV3.

        Parameters
        ----------
        output_layer : str
            Name of the layer to output.
        resize_to : int
            If None, do not resize input images.
            If not None, bilinearly resizes input images to width and height
            `resize_to` before feeding input to model.
            As the network without fully connected layers is fully
            convolutional, it should be able to handle inputs of arbitrary
            size, so resizing might not be strictly needed. If needed, the
            images should be resized to (299, 299).
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1).
        prune : bool
            Whether to free up memory by prune unused layers. (default: False)
        traverse_level : int
            If None, traverse until reached terminal nodes to collect all child
            layers.
            Otherwise, travese up to depth `traverse_level`.

        """
        super(InceptionV3, self).__init__(
            output_layer=output_layer, traverse_level=traverse_level)

        self.resize_to = resize_to
        self.normalize_input = normalize_input

        inception = models.inception_v3(pretrained=True)

        layers = [
            # Block 0: input to maxpool1
            "Conv2d_1a_3x3",
            "Conv2d_2a_3x3",
            "Conv2d_2b_3x3",
            ("MaxPool_1", nn.MaxPool2d(kernel_size=3, stride=2)),

            # Block 1: maxpool1 to maxpool2
            "Conv2d_3b_1x1",
            "Conv2d_4a_3x3",
            ("MaxPool_2", nn.MaxPool2d(kernel_size=3, stride=2)),

            # Block 2: maxpool2 to aux classifier
            "Mixed_5b",
            "Mixed_5c",
            "Mixed_5d",
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",

            # Block 3: aux classifier to final avgpool
            "Mixed_7a",
            "Mixed_7b",
            "Mixed_7c",
            ("AAPool", nn.AdaptiveAvgPool2d(output_size=(1, 1))),
        ]
        for layer in layers:
            if isinstance(layer, str):
                module = getattr(inception, layer)
                self.__setattr__(layer, module)
            else:
                # Pooling layers
                self.__setattr__(layer[0], layer[1])

        # Turn off gradients completely
        for param in self.parameters():
            param.requires_grad = False

        # Prune unused layers
        if prune:
            self._prune_and_update_mapping()

    def forward(self, inp):
        """Get Inception feature maps.

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0.0, 1.0).

        Returns
        -------
        torch.autograd.Variable
            The features extracted from the selected output block.
        """
        outp = []
        x = inp

        if self.resize_to is not None:
            x = F.interpolate(x, size=(self.resize_to, self.resize_to),
                              mode='bilinear', align_corners=False)
        # Scale from range (0, 1) to range (-1, 1)
        if self.normalize_input:
            x = 2 * x - 1
        # Feed-forward
        output = self._forward(inp)

        return output
