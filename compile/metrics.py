"""
All functions in this module must return another function that takes model
predictions and target labels, and return the computed metric.
"""

import torch


def accuracy():
    def acc(output, target):
        """Calculate accuracy given model predictions and target labels."""
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = torch.sum(pred == target).item()
        return correct / len(target)
    return acc


def top_k_accuracy(k=3):
    def top_k_acc(output, target):
        """Calculate top-k accuracy given model predictions and target
        labels."""
        with torch.no_grad():
            pred = torch.topk(output, k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)
    return top_k_acc
