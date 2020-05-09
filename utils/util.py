import math
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import pandas as pd
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """Wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def forward_batch(model, inputs, device, batch_size=128):
    """Forward data in batches and return the stacked results as a torch
    tensor on CPU"""
    num_samples = len(inputs)
    num_batches = math.ceil(num_samples / batch_size)
    outputs = list()

    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size

        output = model(inputs[start:end].to(device=device))
        outputs.append(output.cpu())

    return torch.cat(outputs, dim=0)


class DefaultDict(dict):
    """Custom dictionary implementation with default value or default
    constructor. This also implements `__getattr__` for compatibility and
    convenience."""
    def __init__(self, default_constructor=None, default_value=None, **kwargs):
        """Initialize the custom dict.

        Parameters
        ----------
        default_constructor : callable or None
            If not None, this will be used to generate new value, and the
            `default_value` will be skipped.
        default_value
            Default value of a new key-value pair.
        **kwargs
            Key-value pair(s).

        """
        super(DefaultDict, self).__init__(**kwargs)
        self.default_constructor = default_constructor
        self.default_value = default_value

    def __getitem__(self, key):
        if key not in self:
            if self.default_constructor is not None:
                self[key] = self.default_constructor()
            else:
                self[key] = self.default_value
        return self.get(key)

    def __getattr__(self, name):
        return self[name]


class Cache:
    """Helper class for data caching."""
    def __init__(self):
        return

    def cache(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return


class CustomMetrics:
    """Helper class for dealing with custom metrics.

    Parameters
    ----------
    trainer
        The trainer object.
    metrics
        A single function that takes the state of the trainer and return the
        computed metric, or a list of such functions.

    """
    def __init__(self, trainer, metrics):
        self.trainer = trainer
        if isinstance(metrics, list):
            self.metrics = metrics
        elif metrics is None:
            self.metrics = list()
        else:
            self.metrics = [metrics]
        # Get metric names
        self.metric_names = [met.name for met in self.metrics]
        # Handle metrics which use the same feature extractor
        from compile.metrics_utils.features_extractors \
            import BaseFeatureExtractor
        mets = [met for met in self.metrics if
                isinstance(getattr(met, "model", None), BaseFeatureExtractor)]
        for i, met in enumerate(mets):
            # Compare to all previous metrics
            for j in range(i):
                # If same class and parameters, use the existing feature
                # extractor instead
                if met.model == mets[j].model:
                    met.model = mets[j].model
                    break

    def compute(self):
        computed_metrics = {}
        for i in range(len(self.metrics)):
            metric = self.metrics[i]
            metric_name = self.metric_names[i]
            computed_metrics[metric_name] = metric(self.trainer)
        self.computed_metrics = computed_metrics
        return computed_metrics


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def __getitem__(self, key):
        """Convenience interface to get the current average value of a
        quantity. Identical behavior to `avg`."""
        return self.avg(key)
