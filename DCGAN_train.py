import argparse
import collections

import torch
import numpy as np

import data_loaders.data_loaders as module_data
import models.models as module_arch
import trainers as module_trainers

from parse_config import ConfigParser


def main(config):
    # Fix random seeds for reproducibility
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["seed"])

    logger = config.get_logger("train")

    # Setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data)

    # Build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Get trainer
    trainer = getattr(module_trainers, config["trainer"]["name"])
    trainer = trainer(model=model, data_loader=data_loader, config=config)

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
    config = ConfigParser.from_args(args, options)
    main(config)
