"""Provide the Trainer Class for PyTorch."""

import os
from typing import List

import torch
import torch.utils.tensorboard
import tqdm


# TODO: Log instead of print for epochs + time monitoring
# TODO: Add scheduler ?
# TODO: Load/save
# TODO: Use tensorboard in a better way + Add metrics ?
# TODO: Time info for epochs. (Split data time vs Model time ?)


class PytorchTrainer:
    """Base trainer for pytorch project.

    Wraps all the training procedures for a given model and optimizer.
    """

    train_description = "Training --- loss: {loss:.3f}"
    test_description = "Testing --- loss: {loss:.3f}"
    epoch_description = (
        "Epoch {epoch} --- Avg train loss: {train_loss:.3f} Avg val loss: {val_loss:.3f} [{step}/{total_step}]\n"
    )

    def __init__(self, model, optimizer, device, exp_dir="./experiments", save_mode="never") -> None:
        """Constructor

        Args:
            model (nn.Module): The model to be trained
            optimizer (nn.Optimizer): Optimizer to use
            device (torch.device): Device of the model
            exp_dir (str): Experiment directory, where checkpoints are saved
            save_mode ("never"|"always"|"best"): When to do checkpoints.
                never: No checkpoints are made for this training
                best: Keep only the best checkpoint
                all: A checkpoint is made for each epochs
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.exp_dir = exp_dir
        self.save_mode = save_mode
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir=exp_dir)

        os.makedirs(self.exp_dir, exist_ok=True)
        self.epoch = 0
        self.best_val = float("inf")
        self.best_epoch = -1

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _default_process_batch(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        batch_size = targets.shape[0]
        return inputs, targets, batch_size

    def process_train_batch(self, batch):
        """Process each training batch.

        Extract inputs for the model, targets for the loss, and batch size.

        Should be overwritten for special train batches.

        Args:
            batch (Any): The batch from the train_loader.

        Returns:
            inputs (Any): A valid input for the model
            targets (Any): A valid target for the loss
            batch_size (int): Size of the current batch
        """
        return self._default_process_batch(batch)

    def process_eval_batch(self, batch):
        """Process each eval batch.

        Extract inputs for the model, targets for the loss, and batch size.

        Should be overwritten for special eval batches.

        Args:
            batch (Any): The batch from the eval_loader.

        Returns:
            inputs (Any): A valid input for the model
            targets (Any): A valid target for the loss
            batch_size (int): Size of the current batch
        """
        return self._default_process_batch(batch)

    def backward(self, loss):
        """Backpropagation step given the loss

        Args:
            loss (torch.Tensor)

        Returns:
            loss (torch.Tensor)
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, epochs, train_loader, criterion, val_loader=None, val_criterion=None, epoch_size=0):
        """Train the model

        Args:
            epochs (int): Number of epochs to perform (Should be greater than 0)
            train_loader (iterable): Data loader for the training set
            criterion (callable): Loss function which will be called with the model predictions and the
                targets for each batch. Should return a singled loss value on which to backpropagate
            val_loader (iterable): Optional validation data loader. If provided the model will
                be evaluate after each epoch with it. If not, the validation loss is define as infinite.
            val_criterion (callable): Optional validation loss. Has not to be differentiable. If
                not provided, `criterion` will be used. The lower, the better.
            epoch_size (int): The number of training steps for each epochs.
                By default the length of the train_loader is used.

        Returns:
            Trainer: self
        """
        n_epochs = self.epoch + epochs
        n_steps = 0

        if val_criterion is None:
            val_criterion = criterion

        if epoch_size == 0:
            epoch_size = len(train_loader)

        train_iterator = iter(train_loader)
        while self.epoch < n_epochs:
            self.model.train()
            n_samples = 0
            epoch_loss = 0.0
            loss = float("nan")

            progress = tqdm.trange(epoch_size)
            progress.set_description_str(self.train_description.format(loss=loss))
            for _ in progress:
                n_steps += 1
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                inputs, targets, batch_size = self.process_train_batch(batch)
                predictions = self.model(inputs)
                loss = self.backward(criterion(predictions, targets)).item()

                self.train_losses.append(loss)
                self.tensorboard_writer.add_scalar("Training loss", loss, n_steps)
                n_samples += batch_size
                epoch_loss += loss * batch_size

                progress.set_description_str(self.train_description.format(loss=loss))

            if val_loader is None:
                self.val_losses.append(float("inf"))
            else:
                self.val_losses.append(self.evaluate(val_loader, val_criterion))
                self.tensorboard_writer.add_scalar("Validation loss", self.val_losses[-1], n_steps)

            print(
                self.epoch_description.format(
                    epoch=self.epoch,
                    train_loss=epoch_loss / n_samples,
                    val_loss=self.val_losses[-1],
                    step=self.epoch + 1,
                    total_step=n_epochs,
                ),
                flush=True,
            )

            if self.val_losses[-1] < self.best_val:
                self.best_val = self.val_losses[-1]
                self.best_epoch = self.epoch
                if self.save_mode in ["best", "all"]:
                    self.save("best.pt")

            if self.save_mode == "all":
                self.save(f"{self.epoch}.pt")
            self.epoch += 1

        return self

    def evaluate(self, dataloader, criterion):
        """Evaluate the current model

        Args:
            dataloader (iterable): Dataloader over the dataset to evaluate
            criterion (callable): Evaluation criterion to use

        Returns:
            float: Averaged loss
        """
        self.model.eval()
        with torch.no_grad():
            n_samples = 0
            cum_loss = 0
            loss = float("nan")

            progress = tqdm.tqdm(dataloader)
            progress.set_description_str(self.test_description.format(loss=loss))
            for batch in progress:
                inputs, targets, batch_size = self.process_eval_batch(batch)
                predictions = self.model(inputs)
                loss = criterion(predictions, targets).item()

                n_samples += batch_size
                cum_loss += loss * batch_size

                progress.set_description_str(self.test_description.format(loss=loss))

        return cum_loss / n_samples

    def save(self, filename):
        """Save a checkpoint in the experiment directory

        WIP: For now only save the model

        Args:
            filename (str): Name of the checkpoint file
        """
        torch.save(self.model.state_dict(), os.path.join(self.exp_dir, filename))

    def load(self, path):
        """Not implented yet"""
        raise NotImplementedError
