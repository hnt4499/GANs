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
        Number of fixed noise inputs.
    callbacks : trainers.callbacks
        An early stopping callbacks, which gets called every epoch and return a
        boolean value indicating whether to stop the training process.

    Attributes
    ----------
    fixed_noise
        The fixed noise input tensor, which is fixed and fed into the generator
        to visually see the training progress.
    tracker : utils.MetricTracker
        Used to track the metrics such as loss.
    writer : logger.TensorboardWriter
        A Tensorboard writer.
    config
    data_loader
    model

    """

    def __init__(self, netD, netG, config, data_loader, callbacks=None):
        # Initialize BaseTrainer
        super().__init__(netD, netG, config)
        # Data loader
        self.data_loader = data_loader
        self.len_epoch = len(data_loader)
        # Maximum number of checkpoints to keep
        if "num_ckpt_to_keep" in self.config["trainer"]:
            self.num_ckpt_to_keep = self.config["trainer"]["num_ckpt_to_keep"]
        else:
            self.num_ckpt_to_keep = -1  # keep all
        self.saved_checkpoints = list()  # list of saved checkpoints' paths
        # Fixed noise checkpoint
        self.save_images_every = self.config["trainer"]["save_images_every"]
        # Labels convention
        self.real_label = 1
        self.fake_label = 0
        # Fixed noise input for visual validation
        self.length_z = self.netG.input_length
        self.fixed_noise = torch.randn(
            data_loader.batch_size, self.length_z, 1, 1, device=self.device)
        # Callbacks
        self.callbacks = callbacks
        self.stop = False
        # Metrics tracker. For generator, don't allow any metrics other than
        # loss
        if netD.metric is not None:
            self.keys_D = [m.__name__ + "_D" for m in netD.metric]
        else:
            self.keys_D = list()
        self.tracker = MetricTracker(
            "loss_D", "loss_G", "D_x", "D_G_z1", "D_G_z2",
            *self.keys_D, writer=self.writer)

    def _train_epoch(self):
        """Training logic for an epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
            A log that contains average loss and metrics in this epoch.

        """
        self.netD.train()
        self.netG.train()
        self.tracker.reset()

        for batch_idx, real_data in enumerate(self.data_loader):
            """
            DISCRIMINATOR
            Update discriminator network with all-real batch. Maximize
                log(D(x)) + log(1 - D(G(z)))
            """
            self.netD.zero_grad()
            # Format batch
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            labels_D = torch.full(
                (batch_size,), self.real_label, device=self.device)
            # Forward pass real batch through discriminator
            output_D = self.netD(real_data).view(-1)
            # Calculate loss on all-real batch
            loss_D_real = self.netD.criterion(output_D, labels_D)
            # Calculate gradients for discriminator in backward pass
            loss_D_real.backward()
            # Accuracy of discriminator on all-real batch
            D_x = output_D.mean().item()

            """
            GENERATOR
            Train generator with all-fake batch
            """
            # Input noise
            noise = torch.randn(
                batch_size, self.length_z, 1, 1, device=self.device)
            # Generate fake image batch with generator
            fake_data = self.netG(noise)
            labels_D.fill_(self.fake_label)
            # Classify all-fake batch with discriminator
            output_D = self.netD(fake_data.detach()).view(-1)
            # Calculate discriminator's loss on the all-fake batch
            loss_D_fake = self.netD.criterion(output_D, labels_D)
            # Calculate the gradients for this batch
            loss_D_fake.backward()
            # (1 - accuracy) of discriminator on all-fake data before updating
            D_G_z1 = output_D.mean().item()
            # Add the gradients from the all-real and all-fake batches
            loss_D = loss_D_real + loss_D_fake
            # Update discriminator
            self.netD.optimizer.step()

            """
            GENERATOR
            Update generator network: maximize log(D(G(z)))
            """
            self.netG.zero_grad()
            # Fake labels are real for generator cost
            labels_G = torch.full(
                (batch_size,), self.real_label, device=self.device)
            # Since we just updated discriminator, perform another forward pass
            # of all-fake batch through discriminator
            output_D = self.netD(fake_data).view(-1)
            # Calculate generator's loss based on this output
            loss_G = self.netG.criterion(output_D, labels_G)
            # Calculate gradients for generator
            loss_G.backward()
            # (1 - accuracy) of discriminator on all-fake data after updating
            D_G_z2 = output_D.mean().item()
            # Update generator
            self.netG.optimizer.step()

            # Update logger
            self.writer.set_step((self.epoch - 1) * self.len_epoch + batch_idx)
            # For discriminator
            self.tracker.update("loss_D", loss_D.item())
            self.tracker.update("D_x", D_x)
            self.tracker.update("D_G_z1", D_G_z1)
            self.tracker.update("D_G_z2", D_G_z2)
            for met in self.keys_D:
                self.tracker.update(
                    met.__name__ + "_D", met(output_D, labels_D))
            # For generator, don't allow any metrics other than loss
            self.tracker.update("loss_G", loss_G.item())
            # Print info
            if batch_idx % self.write_logs_every == 0:
                self.logger.debug(
                    "{}{}\tLoss_D: {:.6f}\tLoss_G: {:.6f}\tD(x): {:.4f}\t"
                    "D(G(z)): {:.4f}/{:.4f}".format(
                        self._progress_epoch(self.epoch),
                        self._progress_batch(batch_idx), loss_D.item(),
                        loss_G.item(), D_x, D_G_z1, D_G_z2))

        return self.tracker.result()

    def on_epoch_start(self):
        """Do nothing on epoch start."""
        return

    def on_epoch_end(self):
        """Save images generated from fixed noise inputs as well as model
        checkpoints on epoch end."""
        # Save model state to a checkpoint
        self._save_checkpoint()
        # Cache fake images on fixed noise input
        if self.epoch % self.save_images_every == 0:
            with torch.no_grad():
                fake_data = self.netG(self.fixed_noise).detach().cpu()
            self.writer.add_image(
                "fixed_noise",
                make_grid(fake_data, nrow=8, normalize=True))
        # Early stopping
        if self.callbacks is not None:
            self.stop = self.callbacks(self)

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
