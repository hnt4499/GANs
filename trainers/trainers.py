import os

import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class BaseGANTrainer(BaseTrainer):
    """Training logic for DCGAN.

    Parameters
    ----------
    model
        Instantiated model architecture. A subclass of `base.BaseModel`, found
        in module `models.models`. For example `models.models.DCGAN`.
    data_loader
        Instantiated data loader. A subclass of `base.BaseDataLoader`, defined
        in module `data_loaders.data_loaders`. For example
        `data_loaders.data_loaders.ImageNetLoader`.
    optimizer_D
        Instantiated optimizer for the discriminator, defined in
        `models.optim`, which takes only trainable parameters as arguments.
    criterion_D
        Instantiated loss function for the discriminator, defined in
        `model.loss`, which take only model predictions and target labels, and
        return the computed loss.
    metrics_D
        Instantiated metric function for the discriminator, defined in
        `model.metrics`, which take only model predictions and target labels,
        and return the computed metric.
    optimizer_G
        Instantiated optimizer for the generator, defined in `models.optim`,
        which takes only trainable parameters as arguments.
    criterion_G
        Instantiated loss function for the generator, defined in `model.loss`,
        which take only model predictions and target labels, and return the
        computed loss.
    config
        The configurations parsed from a JSON file.
    num_z : int
        Number of fixed noise inputs.

    Attributes
    ----------
    fixed_noise
        The fixed noise input tensor, which is fixed and fed into the generator
        to visually see the progress.
    tracker
        An object of `utils.MetricTracker`. Used to track the metrics such as
        loss.
    writer
        A Tensorboard writer.
    config
    data_loader
    model

    """

    def __init__(self, model, data_loader, optimizer_D, criterion_D, metrics_D,
                 optimizer_G, criterion_G, config, num_z=64):
        super().__init__(model=model, criterion=None, metric_ftns=None,
                         optimizer=None, config=config)
        self.config = config
        self.data_loader = data_loader
        self.images_every = self.config["trainer"]["images_every"]
        # Don't allow iteration-based training as in original implementation
        self.len_epoch = len(self.data_loader)
        # Don't allow validation
        self.valid_data_loader = None
        self.do_validation = False
        # Maximum number of checkpoints to keep
        if "checkpoint_keep" in self.config["trainer"]:
            self.checkpoint_keep = self.config["trainer"]["checkpoint_keep"]
        else:
            self.checkpoint_keep = -1  # keep all
        self.saved_checkpoints = list()  # list of saved checkpoints
        # Labels convention
        self.real_label = 1
        self.fake_label = 0
        # Fixed noise input for visual validation
        self.length_z = self.model.length_z
        self.num_z = num_z
        self.fixed_noise = torch.randn(
            self.num_z, self.length_z, 1, 1, device=self.device)

        """
        DISCRIMINATOR
        """
        # Loss, metrics and optimizer
        self.criterion_D = criterion_D
        if metrics_D is None:
            self.metrics_D = list()
        else:
            self.metrics_D = metrics_D
        trainable_params_D = filter(
            lambda p: p.requires_grad, model.netD.parameters())
        self.optimizer_D = optimizer_D(trainable_params_D)

        """
        GENERATOR
        """
        # Loss and optimizer
        self.criterion_G = criterion_G
        trainable_params_G = filter(
            lambda p: p.requires_grad, model.netG.parameters())
        self.optimizer_G = optimizer_G(trainable_params_G)

        # Metrics tracker
        # For generator, don't allow any metrics other than loss
        keys_D = [m.__name__ + "_D" for m in self.metrics_D]
        self.tracker = MetricTracker(
            "loss_D", "loss_G", "D_x", "D_G_z1", "D_G_z2",
            *keys_D, writer=self.writer)

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

    def _save_results(self, epoch):
        """Function to be called every at the end of every epochs. Save images
        generate with fixed noise inputs.

        Parameters
        ----------
        epoch : int
            Current epoch number.


        """
        if epoch % self.images_every == 0:
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
        if epoch % self.checkpoint_every == 0:
            # Get model metadata
            model = type(self.model).__name__
            state = {
                "model": model, "epoch": epoch,
                "state_dict_D": self.model.netD.state_dict(),
                "state_dict_G": self.model.netG.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
                "optimizer_G": self.optimizer_G.state_dict(),
                "monitor_best": self.mnt_best,
                "config": self.config.prune(),
            }
            # Save the current model
            filename = str(
                self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            # Cache
            self.saved_checkpoints.append(filename)
            # Remove the oldest checkpoint, if needed
            if self.checkpoint_keep > 0 and \
                    len(self.saved_checkpoints) > self.checkpoint_keep:
                to_remove = self.saved_checkpoints.pop(0)
                os.remove(to_remove)
            # Save the best model if requested
            if save_best:
                best_path = str(self.checkpoint_dir / "model_best.pth")
                torch.save(state, best_path)
                self.logger.info("Saving current best: model_best.pth ...")

    def _load_optimizer(self, checkpoint, optimizer_name):
        """Helper function for loading optimizer's state_dict"""
        # Load optimizer state from checkpoint only when optimizer type is not
        # changed.
        n1 = checkpoint["config"]["trainer"][optimizer_name]["name"]
        n2 = self.config["trainer"][optimizer_name]["name"]
        if n1 != n2:
            self.logger.warning(
                "Warning: Optimizer type given in config file (`{}`) is "
                "different from that of checkpoint (`{}`). Optimizer "
                "parameters not being resumed.".format(n2, n1))
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
        n1 = checkpoint["config"]["trainer"]["model"]["name"]
        n2 = self.config["trainer"]["model"]["name"]
        if n1 != n2:
            self.logger.warning(
                "Warning: Architecture configuration given in config file "
                "(`{}`) is different from that of checkpoint (`{}`). This may "
                "yield an exception while state_dict is being "
                "loaded.".format(n2, n1))
        self.model.netD.load_state_dict(checkpoint["state_dict_D"])
        self.model.netG.load_state_dict(checkpoint["state_dict_G"])

        # Load optimizers
        self._load_optimizer(checkpoint, "optimizer_D")
        self._load_optimizer(checkpoint, "optimizer_G")

        self.logger.info("Checkpoint loaded. Resume training from "
                         "epoch {}".format(self.start_epoch))
