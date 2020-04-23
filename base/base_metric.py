class BaseMetric:
    """Base class for any metric implementation. Subclass must implement
    `__str__`, which is used to print useful logs, and `compute`, which is
    used at the end of every epoch to compute the metric.

    Parameters
    ----------
    name : str
        Name of the metric, for example `acc_D`.

    Attributes
    ----------
    value
        The latest computed metric value.
    name

    """
    def __init__(self, name):
        self.name = name
        self.value = None  # store latest computed value

    def __str__(self):
        raise NotImplementedError

    def __call__(self, trainer):
        raise NotImplementedError
