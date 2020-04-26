class BaseMetric:
    """Base class for any metric implementation. Subclass must implement
    `__str__`, which is used to print useful logs, and `__call__`, which is
    used at the end of every epoch to compute the metric.

    Parameters
    ----------
    name : str
        Name of the metric, for example "acc_D".

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


class FPMetric(BaseMetric):
    """Base class for metrics whose value is floating point number. This base
    class implements default behavior for __str__.

    Parameters
    ----------
    name : str
        Name of the metric, for example "acc_D".
    dp : int
        Number of decimal places to be printed.

    Attributes
    ----------
    value
        The latest computed metric value.
    name
    dp

    """
    def __init__(self, name, dp=4):
        super(FPMetric, self).__init__(name)
        self.dp = dp
        self.repr = "{" + ":.{}f".format(dp) + "}"

    def __str__(self):
        # Format `self.value` to `dp` decimal places
        num = self.repr.format(self.value)
        return "{}: {}".format(self.name, num)
