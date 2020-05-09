import os

import numpy as np
import torch

from base import BaseTrainer
from utils import inf_loop, Cache, CustomMetrics, MetricTracker


class BaseGANTrainer(BaseTrainer):
    """Training logic for DCGAN.

    Parameters
    ----------
    netD
        Instantiated discriminator architecture. A subclass of
        `base.BaseModel`, found in module `models.models`. For example
        `models.models.DCGANDiscriminator`.
    netG
        Instantiated generator architecture. A subclass of `base.BaseModel`,
        found in module `models.models`. For example
        `models.models.DCGANGenerator`.
    metrics
        Instantiated metrics object, defined in `compile.metrics`, which gets
        called every epoch and return the computed metric(s).
    data_loader
        Instantiated data loader. A subclass of `base.BaseDataLoader`, defined
        in module `data_loaders.data_loaders`. For example
        `data_loaders.data_loaders.ImageNetLoader`.
    train_options : dict
        Training options, e.g., the number of epochs.
    config
        The configurations parsed from a JSON file.
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

    def __init__(self, netD, netG, metrics, data_loader, train_options, config,
                 callbacks=None):
        # Initialize BaseTrainer
        super().__init__(netD, netG, train_options, config)
        # Data loader
        self.data_loader = data_loader
        self.len_epoch = len(data_loader)
        # Fixed noise checkpoint
        self.save_images_every = self.train_options["save_images_every"]
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
        # Metrics tracker. Metrics can be a single function or a list of
        # functions. Allow only discriminator to have metrics.
        self.custom_metrics = CustomMetrics(self, metrics)
        self.tracker = MetricTracker(
            "loss_D", "loss_G", "D_x", "D_G_z1", "D_G_z2",
            *self.custom_metrics.metric_names, writer=self.writer)
        # For storing current training information
        self.current_batch = Cache()
        self.global_step = 0

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
            # Batch start
            self.on_batch_start()
            """
            DISCRIMINATOR
            Update discriminator network with all-real batch. Maximize
            log(D(x))
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
            DISCRIMINATOR
            Update discriminator network with all-fake batch. Maximize
            log(1 - D(G(z)))
            """
            # Input noise
            noise = torch.randn(
                batch_size, self.length_z, 1, 1, device=self.device)
            # Generate fake image batch with generator
            generated_from_random_noise = self.netG(noise)
            labels_D.fill_(self.fake_label)
            # Classify all-fake batch with discriminator
            output_D = self.netD(generated_from_random_noise.detach()).view(-1)
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
            output_D = self.netD(generated_from_random_noise).view(-1)
            # Calculate generator's loss based on this output
            loss_G = self.netG.criterion(output_D, labels_G)
            # Calculate gradients for generator
            loss_G.backward()
            # (1 - accuracy) of discriminator on all-fake data after updating
            D_G_z2 = output_D.mean().item()
            # Update generator
            self.netG.optimizer.step()

            # Cache data for future use; and send all data to cpu
            with torch.no_grad():
                generated_from_fixed_noise = self.netG(
                    self.fixed_noise).detach()
            self.current_batch.cache(
                batch_idx=batch_idx,
                batch_size=batch_size,
                global_step=self.epoch * self.len_epoch + batch_idx,

                real_samples=real_data.cpu(),
                fixed_noise=self.fixed_noise.cpu(),
                generated_from_fixed_noise=generated_from_fixed_noise.cpu(),
                random_noise=noise.cpu(),
                generated_from_random_noise=generated_from_random_noise.cpu(),
                output_D=output_D.cpu(),  # disc. output for generated images

                loss_D=loss_D.item(),
                loss_G=loss_G.item(),
                D_x=D_x,
                D_G_z1=D_G_z1,
                D_G_z2=D_G_z2,
            )

            # Batch end
            self.on_batch_end()
            # Early stopping
            if self.stop:
                break

        return self.tracker.result()

    def current(self, key):
        """Helper function to get current batch information"""
        return getattr(self.current_batch, key)

    def on_epoch_start(self):
        """Do nothing on epoch start."""
        return

    def on_epoch_end(self):
        """Do nothing on epoch end."""
        return

    def on_batch_start(self):
        """Do nothing on batch start."""
        return

    def on_batch_end(self):
        """Save images generated from fixed noise inputs as well as model
        checkpoints on batch end."""
        # Update writer and tracker
        self._update_writer()
        self._update_tracker()
        # Write log
        if self.current("global_step") % self.write_logs_every == 0 \
                and self.current("global_step") != 0:
            self._write_log()
        # Save model state to a checkpoint
        if self.current("global_step") % self.save_ckpt_every == 0 \
                and self.current("global_step") != 0:
            filename = "checkpoint-step{}.pth".format(
                self.current("global_step"))
            self._save_checkpoint(
                filename, batch_idx=self.current("batch_idx"),
                global_step=self.current("global_step"))
        # Cache fake images on fixed noise input
        if self.current("global_step") % self.save_images_every == 0 \
                and self.current("global_step") != 0:
            self._write_images_to_tensorboard(
                images=self.current("generated_from_fixed_noise"),
                name="fixed_noise", nrow=8, normalize=True)
        # Early stopping
        if self.callbacks is not None:
            self.stop = self.callbacks(self)

    def _update_writer(self):
        """Update tensorboard writer"""
        if self.writer is not None:
            self.writer.set_step(self.current("global_step"))

    def _update_tracker(self):
        """Update tracker"""
        # For discriminator
        self.tracker.update("loss_D", self.current("loss_D"))
        self.tracker.update("D_x", self.current("D_x"))
        self.tracker.update("D_G_z1", self.current("D_G_z1"))
        self.tracker.update("D_G_z2", self.current("D_G_z2"))
        # For generator, don't allow any metrics other than loss
        self.tracker.update("loss_G", self.current("loss_G"))
        # Compute custom metrics and update to tracker
        custom_metrics = self.custom_metrics.compute()
        for met_name, met in custom_metrics.items():
            self.tracker.update(met_name, met)

    def _write_log(self):
        """Helper function to write logs every `self.write_logs_every`
        batches."""
        # Default info
        to_print = "{}{}\tLoss_D: {:.6f}\tLoss_G: {:.6f}\tD(x): " \
            "{:.4f}\tD(G(z)): {:.4f}/{:.4f}".format(
                self._progress_epoch(self.epoch),
                self._progress_batch(self.current("batch_idx")),
                self.current("loss_D"), self.current("loss_G"),
                self.current("D_x"), self.current("D_G_z1"),
                self.current("D_G_z2"))
        # Add custom metrics info
        for met in self.custom_metrics.metrics:
            to_print += "\t{}".format(str(met))
        self.logger.info(to_print)

    def _get_progess(self, current, total):
        n = len(str(total))
        base = "{:0" + str(n) + "d}"  # e.g "{:04d}"
        base = "[{}/{}]".format(base, base)  # e.g "[{:04d}/{:04d}]"
        return base.format(current, total)

    def _progress_epoch(self, epoch_idx):
        current = epoch_idx
        total = self.epochs - 1
        return self._get_progess(current, total)

    def _progress_batch(self, batch_idx):
        current = batch_idx * self.data_loader.batch_size
        total = self.data_loader.n_samples
        return self._get_progess(current, total)
