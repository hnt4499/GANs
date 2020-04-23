import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


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
        self.metric_names = [met.__name__ for met in self.metrics]

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
