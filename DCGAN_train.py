import argparse
import collections

import torch
import numpy as np

from parse_config import ConfigParser


def main(args, options):
    # Parse config and initialize all objects
    config = ConfigParser.from_args(args, options)
    # Fix random seeds for reproducibility
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["seed"])
    # Get logger
    logger = config.get_logger("train")
    #
    trainer_kwargs = dict()
    # Get outermost instantiated objects and remove all metadata
    t = config["trainer"]["args"]
    for key in t.keys():
        if isinstance(t[key], dict) and "obj" in t[key]:
            trainer_kwargs[key] = t[key]["obj"]
        else:
            trainer_kwargs[key] = t[key]
    # Since `trainer` is ignored, it needs to be initialized
    trainer = getattr(
        config.get_module_from_type(config["trainer"]["type"]),
        config["trainer"]["name"])
    trainer = trainer(config=config, **trainer_kwargs)
    # Start training
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default=None, type=str,
                      help="Config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="Path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="Indices of GPUs to enable (default: all)")

    # Custom cli options to modify configuration from default values given in
    # json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(flags=["--lrd", "--learning_rate_D"], type=float,
                   target="netD;optimizer;args;lr"),
        CustomArgs(flags=["--lrg", "--learning_rate_G"], type=float,
                   target="netG;optimizer;args;lr"),
        CustomArgs(flags=["--bs", "--batch_size"], type=int,
                   target="data_loader;args;batch_size")
    ]
    main(args, options)
