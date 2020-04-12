"""
All functions in this module must return another function or callable class
instance that takes a `trainers.trainers.Trainer` instance and return a boolean
value indicating whether to stop training or not.
"""


class Thresholding:
    """Stop training when a monitored quantity has gone beyond a threshold.

    Parameters
    ----------
    monitor : str
        Quantity to be monitored.
    mode : str
        One of ["min", "max"].
            In `min` mode, monitoring will be started when the quantity
            monitored is higher than the threshold.
            In `max` mode, monitoring will be started when the quantity
            monitored is lower than the threshold.
    threshold : float
        Start monitoring if the quantity monitored go beyond this value.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    patience : int
        Number of steps of no improvements to wait before actually stopping the
        training process.
    verbose : int
        One of [0, 1]. Verbosity mode.

    """
    def __init__(self, monitor, mode, threshold, min_delta=0, patience=0,
                 verbose=0):
        # Check validity
        if mode not in ["min", "max"]:
            raise ValueError('Invalid mode. Expected one of ["min", "max"]. '
                             'Got {} instead.'.format(mode))
        if verbose not in [0, 1]:
            raise ValueError('Invalid verbosity value. Expected one of [0, 1].'
                             ' Got {} instead.'.format(verbose))

        self.monitor = monitor
        self.mode = mode
        self.threshold = threshold

        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose

        self.started = False  # whether monitoring has been started
        self.count = 0  # number of times threshold is exceeded
        self.logger = None  # trainer's logger

    def __call__(self, trainer):
        # Get `monitor` quantity from `utils.MetricTracker` object in the
        # trainer
        value = trainer.tracker[self.monitor]
        # Get logger
        if self.logger is None:
            self.logger = trainer.logger
        # Start monitoring
        if (not self.started) and self._compare(value, self.threshold):
            self.started = True
            self._say(
                "{} has exceeded the threshold. Starting "
                "monitoring...".format(self.monitor))
        # Monitor
        if self.started:
            if self._compare(value, self.threshold):
                self.count += 1
                self._say(
                    "({}/{}) {} exceeds the threshold.".format(
                        self.count, self.patience, self.monitor))
            # Reset counter
            else:
                self.count = 0
                self._say(
                    "{} does not exceed the threshold. Resetting "
                    "counter...".format(self.monitor))
        else:
            self._say(
                "{} does not exceeded the threshold. Monitoring has "
                "not been started.".format(self.monitor))
        # Stop training
        if self.count >= self.patience:
            self._say("Stopping training...")
            return True
        else:
            return False

    def _compare(self, x, y):
        """Helper function to compare whether `y` is improved over `x`, given
        the corresponding `mode` and `delta` value."""
        if self.mode == "max":
            return (y - x - self.min_delta) > 0
        else:
            return (x - y - self.min_delta) > 0

    def _say(self, msg):
        """Helper function to print logs to console."""
        if self.verbose:
            self.logger.info(msg)
