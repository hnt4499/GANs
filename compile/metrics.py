"""
All functions in this module must return another function that takes a trainer
and return the computed metric.
"""

import torch

from base.base_metric import BaseMetric


class Accuracy(BaseMetric):
    """Calculate accuracy of the discriminator classifying generated images.

    Parameters
    ----------
    name : str
        Name of the metric, for example `acc_d`.
    dp : int
        Number of decimal places to be printed. (default: 4)

    Attributes
    ----------
    value
        The latest computed metric value.
    name
    dp

    """
    def __init__(self, name, dp=4):
        super(Accuracy, self).__init__(name)
        self.dp = dp
        self.repr = "{" + ":.{}f".format(dp) + "}"

    def __str__(self):
        # Format `self.value` to `dp` decimal places
        num = self.repr.format(self.value)
        return "{}: {}".format(self.name, num)

    def __call__(self, trainer):
        with torch.no_grad():
            pred = trainer.current_batch.output_D
            pred = (pred >= 0.5).type(torch.float32)
            # Get target labels
            target = torch.full(
                (trainer.current_batch.batch_size,),
                trainer.fake_label, device=trainer.device)
            assert pred.shape[0] == len(target)
            correct = torch.sum(pred == target).item()
        self.value = correct / len(target)
        return self.value
