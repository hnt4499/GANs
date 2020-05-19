import os
import logging
import inspect
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime

from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """Class to parse configuration json file. Handles hyperparameters for
        training, initializations of modules, checkpoint saving and logging
        module.

        Parameters
        ----------
        config : dict
            Dict containing configurations, hyperparameters for training.
            Contents of `config.json` file for example.
        resume : str
            Path to the checkpoint being loaded.
        modification : dict
            Dictionary of keychain:value, specifying position values to be
            replaced from config dict.
        run_id
            Unique Identifier for training processes. Used to save checkpoints
            and training log. Timestamp is being used as default.

        """
        # Load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        # Initialize all objects within the configuration in-place
        self.global_vars = dict()
        self.init_all()

        # Set save_dir where trained model and log will be saved
        save_dir = self.config["trainer"]["args"]["train_options"]["save_dir"]
        if save_dir is None:
            self._checkpoint_dir = None
            self._log_dir = None
        else:
            save_dir = Path(save_dir)
            # Set log details
            exper_name = self.config["name"]
            if run_id is None:  # Use timestamp as default run-id
                run_id = datetime.now().strftime(r"%m%d_%H%M%S")
            dir_name = "{}_{}".format(exper_name, run_id)
            self._checkpoint_dir = save_dir / dir_name / "models"
            self._log_dir = save_dir / dir_name / "logs"
            # Make directory for saving checkpoints and log
            exist_ok = run_id == ""
            self.checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
            # Save updated config file to the checkpoint dir
            write_json(self.prune(), self.checkpoint_dir / "config.json")
        # Configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=""):
        """Initialize this class from some CLI arguments, with an option of
        overwriting some values in the config file. Used in train, test.

        Parameters
        ----------
        args : argparse.ArgumentParser
            An argument parser with some default options such as `config`,
            `resume` and `device`.
        options : list
            A list of dict-like objects whose keys are `flags` (which are used
            for the argument parser), `type` (type of the object) and `target`
            (the path to this flag in the config file, separted by ";", e.g.,
            "trainer;args;models;args;image_size", which controls the input
            image size of the model).

        Returns
        -------
        ConfigParser
            The initialized object.

        """
        # Parse custom options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()
        # Device
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        # If `resume` is not None, resume the training process
        if args.resume is not None:
            # Get the config filepath from the latest checkpoint
            ckpt_path = Path(args.resume)
            config_path = ckpt_path.parent / "config.json"
            # Read the config from the latest checkpoint
            config = read_json(config_path)
            # If `args.config` is specified, overwrite the config from the
            # latest checkpoint with the provided one
            if args.config is not None:
                config.update(read_json(args.config))
        # Otherwise, train from scratch
        else:
            # Raise error if config filepath is not specified
            if args.config is None:
                raise ValueError(
                    "Configuration file need to be specified when training "
                    "from scratch. Add \"-c config.json\", for example.")
            # Read provided config file
            config_path = Path(args.config)
            config = read_json(config_path)

        # Parse custom CLI options into dictionary
        modification = {opt.target: getattr(
            args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, args.resume, modification)

    def __getitem__(self, key):
        """Access items like ordinary dict."""
        return self.config[key]

    def __contains__(self, key):
        """Check if a key is in a dict"""
        return key in self.config

    def get_logger(self, name, verbosity=2):
        """Get logger with a verbosity level."""
        # Check validity
        if verbosity not in self.log_levels:
            raise ValueError(
                "Invalid verbosity option. Expected one of {}, got {} "
                "instead.".format(list(self.log_levels.keys()), verbosity))
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def init_all(self, level=0, path="root", is_arg=False):
        """Recursively initialize all objects within the current configuration
        by modifying it in-place.

        Parameters
        ----------
        config : collections.OrderedDict
            Dict containing configurations, hyperparameters for training.
            Contents of `config.json` file for example.

        """
        # Import modules only when this function is called
        import models as module_models
        import compile as module_compile
        import data_loaders as module_data
        import trainers as module_trainers
        # Mapping for object initialization
        type_mapping = {
            # Models
            "model": module_models.models,
            "weights_init": module_models.weights_init,
            # Data
            "data_loader": module_data.data_loaders,
            "dataset": module_data.datasets,
            "pre_processing": module_data.pre_processing,
            # Compile
            "criterion": module_compile.criterion,
            "metric": module_compile.metrics,
            "features_extractor":
                module_compile.metrics_utils.features_extractors,
            "optimizer": module_compile.optimizers,
            # Trainer
            "trainer": module_trainers.trainers,
            "callbacks": module_trainers.callbacks
        }
        # Save mapping for future use
        self.type_mapping = type_mapping
        # Initialize all objects in-place
        init_all_helper(self.config, self.type_mapping, self.global_vars)

    def prune(self):
        """Recursively remove all instantiated objects from a configuration
        dictionary and return a new one with only metadata left.

        Parameters
        ----------
        config : dict-like
            Dict containing configurations, hyperparameters for training.
            Contents of `config.json` file for example.
        """
        return _prune_helper(self.config)

    def get_module_from_type(self, type):
        """Get correct module from type specified in the configuration. For
        example, get_module_from_type("metric") will return the module
        `compile.metrics`.

        Parameters
        ----------
        type : str
            Name of the module used in the configuration.
        """
        if type not in self.type_mapping:
            raise ValueError(
                "Invalid type. Expected one of {}, got {} "
                "instead.".format(list(self.type_mapping.keys()), type))
        return self.type_mapping[type]

    # Setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def log_dir(self):
        return self._log_dir


def _prune_helper(config):
    """Recursion helper function for `ConfigParser.prune`"""
    if not isinstance(config, dict):
        return config

    keys = config.keys()
    new_config = dict()
    for key in keys:
        # If "obj", ignore it
        if key == "obj":
            continue
        new_config[key] = _prune_helper(config[key])
    return new_config


"""Helper functions to update config dict with custom CLI options"""


def _update_config(config, modification):
    """Update current config with custom CLI options."""
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    # Get sequence of keys
    keys = keys.split(";")
    # Traverse down to the second-last key
    parent = reduce(getitem, keys[:-1], tree)
    # Set the value for the last key
    parent[key[-1]] = value


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")


"""Helper functions to deal with object initializations"""


def _get_kwargs(kwargs):
    """Helper function to prune outermost instantiated objects' metadata
    and return the clean keyword arguments."""
    clean_kwargs = dict()
    for key in kwargs.keys():
        if isinstance(kwargs[key], dict) and "obj" in kwargs[key]:
            clean_kwargs[key] = kwargs[key]["obj"]
        else:
            clean_kwargs[key] = kwargs[key]
    return clean_kwargs


def _get_initialized_object(config, path, type_mapping, global_vars):
    """Helper function to initialize object given "type", "name" and "args"."""
    # Check validity
    if config["type"] not in type_mapping:
        raise ValueError(
            "Invalid type at {}. Expected one of {}, got \"{}\" "
            "instead".format(path, list(type_mapping.keys()), config["type"]))
    # Initialize
    kwargs = _get_kwargs(config["args"])
    fn = getattr(type_mapping[config["type"]], config["name"])
    # Append global variables
    all_kwargs = inspect.getfullargspec(fn)[0]
    for kwarg in all_kwargs:
        if kwarg not in kwargs and kwarg in global_vars:
            kwargs[kwarg] = global_vars[kwarg]
    return fn(**kwargs)


def init_all_helper(config, type_mapping, global_vars, level=0, path="root",
                    is_arg=False):
    """Recursion helper function for `ConfigParser.init_all`.

    Parameters
    ----------
    config : dict-like
    type_mapping : dict
        Used to map a "type" with its corresponding module.
    level : int
        Current recursion depth. Used for any informative error raised.
    path : str
        Path to current node. Used for any informative error raised.
    is_arg : bool
        Used to indicate whether the current node is an argument of a function.

    """
    if not isinstance(config, dict):
        return
    # Parse global variables
    if level == 0 and "global" in config:
        for key, value in config["global"].items():
            global_vars[key] = value
    keys = config.keys()
    # If any of ["type", "name", "args"] is in config.keys() and the others
    # are not, raise an Error
    isin = [key in keys for key in ["type", "name", "args"]]
    if (not is_arg) and level > 0 and any(isin):
        if sum(isin) < 3:
            raise ValueError(
                'One or more of ["type", "name", "args"] are missing at '
                + path)
        # Initialize children objects
        for key in keys:
            init_all_helper(
                config[key], type_mapping, global_vars, level + 1, path,
                is_arg=True)

        # Ignore current object if `ignored` is set to True
        if "ignored" in config and config["ignored"]:
            return
        # Otherwise, initialize current object
        config["obj"] = _get_initialized_object(
            config, path, type_mapping, global_vars)
    # Allow function arguments to have one of ["type", "name", "args"]
    else:
        # Handle object as a list of multiple objects.
        if "0" in keys:
            # Get valid keys
            i = 0
            while str(i) in keys:
                i += 1
            keys = list(map(str, range(i)))
        # Initialize objects
        for key in keys:
            child_path = path + "->" + key
            init_all_helper(
                config[key], type_mapping, global_vars, level + 1, child_path,
                is_arg=False)
        # Post-handle multiple objects for compatibility
        if "0" in keys:
            objs = list()
            for key in keys:
                if isinstance(config[key], dict) and "obj" in config[key]:
                    objs.append(config[key]["obj"])
                    config[key].pop("obj", None)
            config["obj"] = objs
