"""
All functions in this module must return another function that takes a trainer
and return the computed metric.
"""

import torch


def accuracy():
    def acc_D(trainer):
        """Calculate accuracy given model predictions and target labels of
        generated samples."""
        with torch.no_grad():
            # Get predictions
            pred = trainer.current_batch.output_D
            pred = (pred >= 0.5).type(torch.float32)
            # Get target labels
            target = torch.full(
                (trainer.current_batch.batch_size,),
                trainer.fake_label, device=trainer.device)
            assert pred.shape[0] == len(target)
            correct = torch.sum(pred == target).item()
        return correct / len(target)
    return acc_D
