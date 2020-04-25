import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 for features extraction."""

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self, output_block=3, resize_to=None, normalize_input=True):
        """Build pretrained InceptionV3.

        Parameters
        ----------
        output_block : int
            Index of block to return features of (default: 3). Possible value
            is:
                0: corresponds to output of first max pooling,
                1: corresponds to output of second max pooling,
                2: corresponds to output which is fed to aux classifier,
                3: corresponds to output of final average pooling.
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
        """
        super(InceptionV3, self).__init__()

        if output_block not in self.BLOCK_INDEX_BY_DIM.values():
            raise ValueError(
                "Invalid index of output block. Expected one of {}, got {} "
                "instead.".format(
                    list(self.BLOCK_INDEX_BY_DIM.values()), output_block))

        self.output_block = output_block
        self.resize_to = resize_to
        self.normalize_input = normalize_input

        inception = models.inception_v3(pretrained=True)
        self.blocks = nn.ModuleList()

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.output_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.output_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.output_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        # Turn off gradients completely
        for param in self.parameters():
            param.requires_grad = False

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
        for idx in range(self.output_block + 1):
            x = self.blocks[idx](x)

        return x
