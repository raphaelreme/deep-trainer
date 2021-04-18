"""Provide the Trainer Class for PyTorch."""

import os
from typing import Callable, Dict, List

import torch
import torch.utils.tensorboard
import tqdm

from . import metric


# TODO: Log instead of print for epochs + time monitoring
# TODO: Add scheduler ?
# TODO: Load/save
# TODO: Add some more stuff to tensorboard ?
# TODO: Time info for epochs. (Split data time vs Model time ?)


def _convert_to_deep_trainer_criterion(criterion: Callable) -> metric.Criterion:
    if not isinstance(criterion, metric.Criterion):
        return metric.AveragingCriterion(criterion)
    return criterion


class PytorchTrainer:
    """Base trainer for pytorch project.

    Wraps all the training procedures for a given model and optimizer.
    """

    train_description = "Training --- loss: {loss:.3f}"
    test_description = "Testing --- loss: {loss:.3f}"
    epoch_description = (
        "Epoch {epoch} --- Avg train loss: {train_loss:.3f} Avg val loss: {val_loss:.3f} [{step}/{total_step}]\n"
    )

    def __init__(self, model, optimizer, device, exp_dir="./experiments", save_mode="never"):
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
        self.train_steps = 0
        self.epoch = 0
        self.best_val = float("inf")
        self.best_epoch = -1

        self.train_losses: List[float] = []
        self.val_losses: Dict[str, List[float]] = {}

    def _default_process_batch(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return inputs, targets

    def process_train_batch(self, batch):
        """Process each training batch.

        Extract inputs for the model, targets for the loss, and batch size.

        Should be overwritten for special train batches.

        Args:
            batch (Any): The batch from the train_loader.

        Returns:
            inputs (Any): A valid input for the model
            targets (Any): A valid target for the loss
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

    def train(self, epochs, train_loader, criterion, val_loader=None, val_criteria=None, epoch_size=0):
        """Train the model

        Args:
            epochs (int): Number of epochs to perform (Should be greater than 0)
            train_loader (iterable): Data loader for the training set
            criterion (callable): Loss function which will be called with the model predictions and the
                targets for each batch. Should return a singled loss value on which to backpropagate.
                It should be differentiable.
            val_loader (iterable): Optional validation data loader. If provided the model will
                be evaluate after each epoch with it. If not, the validation loss is define as infinite.
            val_criteria (List[callable]): Optional validation metrics. If not provided, `criterion` will
                be used. The first one will be used to select the best model which minimizes the criterion.
            epoch_size (int): The number of training steps for each epochs.
                By default the length of the train_loader is used.

        Returns:
            Trainer: self
        """
        n_epochs = self.epoch + epochs

        if epoch_size == 0:
            epoch_size = len(train_loader)

        criterion = _convert_to_deep_trainer_criterion(criterion)

        if val_criteria is None:
            val_criteria = [criterion]

        train_iterator = iter(train_loader)
        while self.epoch < n_epochs:
            self.model.train()
            loss = float("nan")
            criterion.reset()

            progress = tqdm.trange(epoch_size)
            progress.set_description_str(self.train_description.format(loss=loss))
            for _ in progress:
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                inputs, targets = self.process_train_batch(batch)
                predictions = self.model(inputs)
                loss = self.backward(criterion(predictions, targets)).item()

                self.train_losses.append(loss)
                self.tensorboard_writer.add_scalar("Training loss", loss, self.train_steps)

                progress.set_description_str(self.train_description.format(loss=loss))
                self.train_steps += 1

            if val_loader is None:
                for key in self.val_losses:
                    self.val_losses[key].append(float("nan"))
            else:
                losses = self.evaluate(val_loader, val_criteria)
                for key in set(losses).union(self.val_losses):
                    if key not in self.val_losses:
                        self.val_losses[key] = [float("nan")] * self.epoch
                    self.val_losses[key].append(losses.get(key, float("nan")))

                    self.tensorboard_writer.add_scalar(key, self.val_losses[key][-1], self.train_steps)

            val_loss = self.val_losses[val_criteria[0].name][-1]

            print(
                self.epoch_description.format(
                    epoch=self.epoch,
                    train_loss=criterion.aggregate(),
                    val_loss=val_loss,
                    step=self.epoch + 1,
                    total_step=n_epochs,
                ),
                flush=True,
            )

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_epoch = self.epoch
                if self.save_mode in ["best", "all"]:
                    self.save("best.pt")

            if self.save_mode == "all":
                self.save(f"{self.epoch}.pt")

            self.epoch += 1

        return self

    def evaluate(self, dataloader, criteria):
        """Evaluate the current model on several criteria

        Args:
            dataloader (iterable): Dataloader over the dataset to evaluate
            criteria (List[callable]): Evaluation criterion to use

        Returns:
            Dict[str, float]: Aggregate loss for each criteria
        """
        if len(criteria) == 0:
            return []

        criteria = list(map(_convert_to_deep_trainer_criterion, criteria))
        for criterion in criteria:
            criterion.reset()

        self.model.eval()
        with torch.no_grad():
            loss = float("nan")

            progress = tqdm.tqdm(dataloader)
            progress.set_description_str(self.test_description.format(loss=loss))
            for batch in progress:
                inputs, targets = self.process_eval_batch(batch)
                predictions = self.model(inputs)

                for criterion in reversed(criteria):
                    loss = criterion(predictions, targets).item()

                progress.set_description_str(self.test_description.format(loss=loss))

        return {criterion.name: criterion.aggregate() for criterion in criteria}

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
