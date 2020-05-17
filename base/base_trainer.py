import os
from queue import Queue
from abc import abstractmethod

import torch
from torchvision.utils import make_grid
from numpy import inf

from logger import TensorboardWriter
from utils import DefaultDict


class BaseTrainer:
    """
    Base class for all trainers, taking care of basic parameters and
    configurations.
    """
    def __init__(self, netD, netG, train_options, config):

        # Cache data
        self.netD = netD
        self.netG = netG
        self.train_options = DefaultDict(default_value=None, **train_options)
        self.config = config
        self.stop = False  # use for early stopping
        # Setup GPU device if available and move model into configured device
        self.device, self.device_ids = self._get_device(config["n_gpu"])
        self.netD = self.netD.to(self.device)
        self.netG = self.netG.to(self.device)
        if len(self.device_ids) > 1:
            self.netD = torch.nn.DataParallel(
                self.netD, device_ids=self.device_ids)
            self.netG = torch.nn.DataParallel(
                self.netG, device_ids=self.device_ids)

        # Training configuration
        self.epochs = self.train_options["epochs"]
        self.save_ckpt_every = self.train_options["save_ckpt_every"]
        self.write_logs_every = self.train_options["write_logs_every"]
        self.evaluate_every = self.train_options["evaluate_every"]
        # Directory to save models and logs
        self.checkpoint_dir = config.checkpoint_dir
        self.log_dir = config.log_dir
        # Maximum number of checkpoints to keep
        self.num_ckpt_to_keep = self.train_options["num_ckpt_to_keep"]
        self.saved_checkpoints = Queue(maxsize=0)  # checkpoint paths

        # Get logger
        self.logger = config.get_logger(
            "trainer", verbosity=self.train_options["verbosity"])
        # Setup visualization writer instance
        if self.log_dir is None:
            self.writer = None
        else:
            self.writer = TensorboardWriter(
                self.log_dir, self.logger,
                enabled=self.train_options["tensorboard"])
        # Resume checkpoint if specified
        if "resume" in config and config.resume is not None:
            self._resume_checkpoint(config.resume)
        else:
            self.start_epoch = 0

    def _get_device(self, n_gpu_use):
        """
        Setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\"s no GPU available on this "
                                "machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\"s configured to "
                                "use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, filename=None, **custom_info):
        """Save current model state to a checkpoint.

        Parameters
        ----------
        filename : str
            Saved file name. If None, use default filename format.
        **custom_info
            Custom information to add to the state dictionary.

        """
        if self.checkpoint_dir is None:
            return
        # State dictionary
        state = {
            "epoch": self.epoch,
            "config": self.config.prune(),

            "netD_name": type(self.netD).__name__,
            "netG_name": type(self.netG).__name__,
            "netD_state_dict": self.netD.state_dict(),
            "netG_state_dict": self.netG.state_dict(),

            "optimD_name": type(self.netD.optimizer).__name__,
            "optimG_name": type(self.netG.optimizer).__name__,
            "optimD_state_dict": self.netD.optimizer.state_dict(),
            "optimG_state_dict": self.netG.optimizer.state_dict(),
        }
        state.update(custom_info)
        # Save checkpoint
        if filename is None:
            filename = "checkpoint-epoch{}.pth".format(self.epoch)
        filepath = self.checkpoint_dir / filename
        torch.save(state, filepath)
        self.saved_checkpoints.put(filepath)
        self.logger.info("Saving checkpoint: {} ...".format(filepath))
        # Remove old checkpoint
        if self.num_ckpt_to_keep is not None and \
                self.saved_checkpoints.qsize() > self.num_ckpt_to_keep:
            old_ckpt = self.saved_checkpoints.get()
            os.remove(old_ckpt)
            # Make sure it works as expected
            assert self.saved_checkpoints.qsize() == self.num_ckpt_to_keep

    def _resume_checkpoint(self, resume_path):
        """Resume from saved checkpoints.

        Parameters
        ----------
        resume_path : str
            Checkpoint path to be resumed.

        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        # Load architecture params from checkpoint.
        curr_netD = type(self.netD).__name__
        curr_netG = type(self.netG).__name__
        if (checkpoint["netD_name"] != curr_netD) or \
                (checkpoint["netG_name"] != curr_netG):
            self.logger.warning(
                "Warning: Architecture configuration given in config file is "
                "different from that of checkpoint. This may yield an "
                "exception while state_dict is being loaded.")
        self.netD.load_state_dict(checkpoint["state_dict_D"])
        self.netG.load_state_dict(checkpoint["state_dict_G"])
        # Load optimizer state from checkpoint only when optimizer type is not
        # changed.
        curr_optimD = type(self.netD.optimizer).__name__
        curr_optimG = type(self.netG.optimizer).__name__
        if (checkpoint["optimD_name"] != curr_optimD) or \
                (checkpoint["optimG_name"] != curr_optimG):
            self.logger.warning(
                "Warning: Optimizer type given in config file is different "
                "from that of checkpoint. Optimizer parameters not being "
                "resumed.")
        else:
            self.netD.optimizer.load_state_dict(
                checkpoint["optimD_state_dict"])
            self.netG.optimizer.load_state_dict(
                checkpoint["optimG_state_dict"])

        self.logger.info("Checkpoint loaded. Resume training from "
                         "epoch {}".format(self.start_epoch))

    def _write_images_to_tensorboard(self, images, name, **kwargs):
        """Write images to tensorboard writer.

        Parameters
        ----------
        name : str
            Name to be displayed in tensorboard.
        images : torch.Tensor
            Images to write.
        **kwargs
            Keyword arguments to pass to `make_grid` function.

        """
        if self.writer is not None:
            self.writer.add_image(name, make_grid(images, **kwargs))

    @abstractmethod
    def _train_epoch(self):
        """Training logic for an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
            A `dict` containing computed loss and metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def on_epoch_start(self):
        """
        Function to be called at the beginning of every epoch. Note that this
        function takes no arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self):
        """
        Function to be called at the end of every epoch. Note that this
        function takes no arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def on_batch_start(self):
        """
        Function to be called at the beginning of every batch. Note that this
        function takes no arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def on_batch_end(self):
        """
        Function to be called at the end of every batch. Note that this
        function takes no arguments.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            # Epoch start
            self.on_epoch_start()
            # Start training
            self._train_epoch()
            # Epoch end
            self.on_epoch_end()
            if self.stop:
                break
