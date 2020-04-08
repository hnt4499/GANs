import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker

import models.loss as module_loss
import models.metrics as module_metrics


class BaseGANTrainer(BaseTrainer):
    """Training logic for DCGAN.

    Parameters
    ----------
    model
        Instantiated model architecture. A subclass of `base.BaseModel`, found
        in module `model.model`. For example `model.model.DCGAN`.
    data_loader
        Instantiated data loader. A subclass of `base.BaseDataLoader`, found in
        module `data_loader.data_loaders`.
        For example `data_loader.data_loaders.ImageNetLoader`.
    config : type
        Description of parameter `config`.

    Attributes
    ----------
    fixed_noise : type
        Description of attribute `fixed_noise`.
    tracker : type
        Description of attribute `tracker`.
    writer : type
        Description of attribute `writer`.
    config
    data_loader
    model

    """

    def __init__(self, model, data_loader, config):
        super().__init__(model=model, criterion=None, metric_ftns=None,
                         optimizer=None, config=config)
        self.config = config
        self.data_loader = data_loader
        # Don't allow iteration-based training as in original implementation
        self.len_epoch = len(self.data_loader)
        # Don't allow validation
        self.valid_data_loader = None
        self.do_validation = False
        # Labels convention
        self.real_label = 1
        self.fake_label = 0
        # Fixed noise input for visual validation
        self.length_z = self.model.length_z
        self.num_z = self.config["compile"]["netG"]["num_z"]
        self.fixed_noise = torch.randn(
            self.num_z, self.length_z, 1, 1, device=self.device)

        """
        DISCRIMINATOR
        """
        # Loss and metrics
        self.criterion_D = getattr(
            module_loss, config["compile"]["netD"]["loss"])
        if config["compile"]["netD"]["metrics"] is None:
            self.metrics_D = list()
        else:
            self.metrics_D = [getattr(module_metrics, met) for met in
                              config["compile"]["netD"]["metrics"]]
        # Initialize optimizer and learning rate scheduler for discriminator
        trainable_params_D = filter(
            lambda p: p.requires_grad, model.netD.parameters())
        self.optimizer_D = self._init_obj(
            "netD", "optimizer", torch.optim, trainable_params_D)
        self.lr_scheduler_D = self._init_obj(
            "netD", "lr_scheduler", torch.optim.lr_scheduler, self.optimizer_D)

        """
        GENERATOR
        """
        self.criterion_G = getattr(
            module_loss, config["compile"]["netG"]["loss"])
        # Initialize optimizer and learning rate scheduler for discriminator
        trainable_params_G = filter(
            lambda p: p.requires_grad, model.netG.parameters())
        self.optimizer_G = self._init_obj(
            "netG", "optimizer", torch.optim, trainable_params_G)
        self.lr_scheduler_G = self._init_obj(
            "netG", "lr_scheduler", torch.optim.lr_scheduler, self.optimizer_G)

        # Metrics tracker
        # For generator, don't allow any metrics other than loss
        keys_D = [m.__name__ + "_D" for m in self.metrics_D]
        self.tracker = MetricTracker(
            "loss_D", "loss_G", "D_x", "D_G_z1", "D_G_z2",
            *keys_D, writer=self.writer)

    def _init_obj(self, m, name, module, *args, **kwargs):
        """Helper function to initialize objects for Discriminator and
        Generator."""
        if self.config["compile"][m][name] is None:
            return None
        module_name = self.config["compile"][m][name]["type"]
        module_args = dict(self.config["compile"][m][name]["args"])
        assert all([k not in module_args for k in kwargs]), \
            "Overwriting kwargs given in config file is not allowed."
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def _train_epoch(self, epoch):
        """Training logic for an epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
            A log that contains average loss and metrics in this epoch.

        """
        self.model.netD.train()
        self.model.netG.train()
        self.tracker.reset()

        for batch_idx, real_data in enumerate(self.data_loader):
            """
            DISCRIMINATOR
            Update discriminator network with all-real batch. Maximize
                log(D(x)) + log(1 - D(G(z)))
            """
            self.model.netD.zero_grad()
            # Format batch
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            labels_D = torch.full(
                (batch_size,), self.real_label, device=self.device)
            # Forward pass real batch through discriminator
            output_D = self.model.netD(real_data).view(-1)
            # Calculate loss on all-real batch
            loss_D_real = self.criterion_D(output_D, labels_D)
            # Calculate gradients for discriminator in backward pass
            loss_D_real.backward()
            # Accuracy of discriminator on all-real batch
            D_x = output_D.mean().item()

            """
            GENERATOR
            Train generator with all-fake batch
            """
            self.model.netG.zero_grad()
            # Input noise
            noise = torch.randn(
                batch_size, self.length_z, 1, 1, device=self.device)
            # Generate fake image batch with generator
            fake_data = self.model.netG(noise)
            labels_D.fill_(self.fake_label)
            # Classify all-fake batch with discriminator
            output_D = self.model.netD(fake_data.detach()).view(-1)
            # Calculate discriminator's loss on the all-fake batch
            loss_D_fake = self.criterion_D(output_D, labels_D)
            # Calculate the gradients for this batch
            loss_D_fake.backward()
            # (1 - accuracy) of discriminator on all-fake data before updating
            D_G_z1 = output_D.mean().item()
            # Add the gradients from the all-real and all-fake batches
            loss_D = loss_D_real + loss_D_fake
            # Update discriminator
            self.optimizer_D.step()

            """
            GENERATOR
            Update generator network: maximize log(D(G(z)))
            """
            self.model.netG.zero_grad()
            # Fake labels are real for generator cost
            labels_G = torch.full(
                (batch_size,), self.real_label, device=self.device)
            # Since we just updated discriminator, perform another forward pass
            # of all-fake batch through discriminator
            output_D = self.model.netD(fake_data).view(-1)
            # Calculate generator's loss based on this output
            loss_G = self.criterion_G(output_D, labels_G)
            # Calculate gradients for generator
            loss_G.backward()
            # (1 - accuracy) of discriminator on all-fake data after updating
            D_G_z2 = output_D.mean().item()
            # Update generator
            self.optimizer_G.step()

            # Update logger
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # For discriminator
            self.tracker.update("loss_D", loss_D.item())
            self.tracker.update("D_x", D_x)
            self.tracker.update("D_G_z1", D_G_z1)
            self.tracker.update("D_G_z2", D_G_z2)
            for met in self.metrics_D:
                self.tracker.update(
                    met.__name__ + "_D", met(output_D, labels_D))
            # For generator, don't allow any metrics other than loss
            self.tracker.update("loss_G", loss_G.item())

            # Print info and save images
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "{}{}\tLoss_D: {:.6f}\tLoss_G: {:.6f}\tD(x): {:.4f}\t"
                    "D(G(z)): {:.4f}/{:.4f}".format(
                        self._progress_epoch(epoch),
                        self._progress_batch(batch_idx), loss_D.item(),
                        loss_G.item(), D_x, D_G_z1, D_G_z2))

        # Cache fake images on fixed noise input
        with torch.no_grad():
            fake_data_fixed = self.model.netG(
                self.fixed_noise).detach().cpu()
            self.fake_data_fixed = fake_data_fixed

        # Update learning rate
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()

        return self.tracker.result()

    def _get_progess(self, current, total):
        n = len(str(total))
        base = "{:0" + str(n) + "d}"  # e.g "{:04d}"
        base = "[{}/{}]".format(base, base)  # e.g "[{:04d}/{:04d}]"
        return base.format(current, total)

    def _progress_epoch(self, epoch_idx):
        current = epoch_idx
        total = self.epochs
        return self._get_progess(current, total)

    def _progress_batch(self, batch_idx):
        current = batch_idx * self.data_loader.batch_size
        total = self.data_loader.n_samples
        return self._get_progess(current, total)

    def _save_results(self):
        self.writer.add_image(
            "fixed_noise",
            make_grid(self.fake_data_fixed, nrow=8, normalize=True))

    def _save_checkpoint(self, epoch, save_best=False):
        """Saving checkpoints.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        save_best : bool
            If True, rename the saved checkpoint to "model_best.pth".

        """
        arch = type(self.model).__name__
        state = {
            "arch": arch, "epoch": epoch,
            "state_dict_D": self.model.netD.state_dict(),
            "state_dict_G": self.model.netG.state_dict(),
            "monitor_best": self.mnt_best, "config": self.config,
            "optimizer_D": self.optimizer_D.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(
            epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _load_optimizer(self, checkpoint, optimizer_name):
        """Helper function for loading optimizer's state_dict"""
        # Load optimizer state from checkpoint only when optimizer type is not
        # changed.
        if checkpoint["config"][optimizer_name]["type"] != \
                self.config[optimizer_name]["type"]:

            self.logger.warning(
                "Warning: Optimizer type given in config file is different "
                "from that of checkpoint. Optimizer parameters not being "
                "resumed.")
        else:
            optimizer = getattr(self, optimizer_name)
            optimizer.load_state_dict(checkpoint[optimizer_name])

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
        self.mnt_best = checkpoint["monitor_best"]

        # Load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is "
                "different from that of checkpoint. This may yield an "
                "exception while state_dict is being loaded.")
        self.model.netD.load_state_dict(checkpoint["state_dict_D"])
        self.model.netG.load_state_dict(checkpoint["state_dict_G"])

        # Load optimizers
        self._load_optimizer(checkpoint, "optimizer_D")
        self._load_optimizer(checkpoint, "optimizer_G")

        self.logger.info("Checkpoint loaded. Resume training from "
                         "epoch {}".format(self.start_epoch))
