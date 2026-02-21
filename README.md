# Deep-Trainer

[![License](https://img.shields.io/github/license/raphaelreme/deep-trainer)](https://github.com/raphaelreme/deep-trainer/raw/main/LICENSE)
[![PyPi](https://img.shields.io/pypi/v/deep-trainer)](https://pypi.org/project/deep-trainer)
[![Python](https://img.shields.io/pypi/pyversions/deep-trainer)](https://pypi.org/project/deep-trainer)
[![Downloads](https://img.shields.io/pypi/dm/deep-trainer)](https://pypi.org/project/deep-trainer)
[![Codecov](https://codecov.io/github/raphaelreme/deep-trainer/graph/badge.svg)](https://codecov.io/github/raphaelreme/deep-trainer)
[![Lint and Test](https://github.com/raphaelreme/deep-trainer/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/deep-trainer/actions/workflows/tests.yml)

Lightweight training utilities for PyTorch projects.

`deep-trainer` provides a minimal yet flexible training loop abstraction
for PyTorch projects, including:

- Training & evaluation loops
- Automatic Mixed Precision (AMP) support
- Checkpointing (best / last / all)
- Metric handling system with aggregation
- TensorBoard logging (or custom loggers)
- Easy subclassing for custom training behavior

------------------------------------------------------------------------

## ⚠️ Project Status

This project was originally developed as a personal baseline training
framework.

-   The codebase is **functional but relatively old**
-   APIs may evolve in future versions
-   Some refactoring and cleanup are planned
-   Backward compatibility is not guaranteed for future major
    updates

If you use this project in production or research, please consider
pinning a version.

Contributions and improvements are welcome.

------------------------------------------------------------------------

## 🚀 Installation

### Install with pip

``` bash
pip install deep-trainer
```

### Install from source

``` bash
git clone https://github.com/raphaelreme/deep-trainer.git
cd deep-trainer
pip install .
```

------------------------------------------------------------------------

## 🏁 Getting Started

Below is a minimal training example for a classification task.

``` python
import torch
from deep_trainer import PytorchTrainer

# ======================
# Dataset
# ======================

trainset = ...
valset = ...
testset = ...

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=256)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256)

# ======================
# Model
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ...
model.to(device)

# ======================
# Optimizer & Scheduler
# ======================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Step every batch (scheduler is stepped per training step)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=len(train_loader) * 50,  # decay every 50 epochs
    gamma=0.1,
)

# ======================
# Loss
# ======================

criterion = torch.nn.CrossEntropyLoss()

# ======================
# Training
# ======================

trainer = PytorchTrainer(
    model,
    optimizer,
    scheduler=scheduler,
    save_mode="small",   # keep best + last checkpoint
    device=device,
    use_amp=True,        # optional mixed precision
)

trainer.train(
    epochs=150,
    train_loader=train_loader,
    criterion=criterion,
    val_loader=val_loader,
)

# ======================
# Testing (Best model)
# ======================

trainer.load("experiments/checkpoints/best.ckpt")  # Reload best checkpoint
test_metrics = trainer.evaluate(test_loader)

print(test_metrics)
```

------------------------------------------------------------------------

## Features Overview

### ✔ Simple Trainer Abstraction

The `PytorchTrainer` handles:

-   Forward / backward passes
-   Optimizer and scheduler stepping
-   Mixed precision scaling
-   Metric tracking
-   Validation & best checkpoint selection
-   Logging

You probably will need to override the following method:

-   `process_train_batch`
-   `train_step`
-   `backward`
-   `eval_step`

to customize behavior (multi-loss, gradient accumulation, multiple
optimizers, self-supervised learning, etc.).

------------------------------------------------------------------------

### ✔ Flexible Metric System

The metric system supports:

-   Per-batch metrics
-   Aggregated metrics
-   Validation metric selection
-   Custom metrics via subclassing

------------------------------------------------------------------------

### ✔ Logging

By default, logs are written to TensorBoard.

``` bash
tensorboard --logdir experiments/logs/
```

You can also use:

-   `DictLogger` (in-memory logging)
-   `MultiLogger` (combine multiple loggers)
-   Or implement your own logger by subclassing `TrainLogger`

------------------------------------------------------------------------

## Example

An example training script is available in:

    example/example.py

It demonstrates training a **PreActResNet18** on CIFAR-10.

To use it:

``` bash
# Show available hyperparameters
python example.py -h

# Launch training
python example.py

# Monitor training
tensorboard --logdir experiments/logs/
```

With default parameters, it reaches approximately **94--95% validation
accuracy** on CIFAR-10.

------------------------------------------------------------------------

## Design Philosophy

`deep-trainer` aims to be:

-   Minimal (no heavy abstractions)
-   Transparent (easy to read & debug)
-   Hackable (easy to override core behavior)
-   Suitable for research baselines

It is **not** intended to replace full-featured training frameworks
like:
- PyTorch Lightning
- HuggingFace Trainer
- Accelerate

Instead, it provides a lightweight middle ground between raw PyTorch
loops and larger ecosystems.

------------------------------------------------------------------------

## Contributing

Contributions are welcome!

If you'd like to:

-   Improve documentation
-   Refactor old components
-   Add new metrics
-   Improve testing
-   Modernize APIs

Feel free to open an issue or submit a pull request.

------------------------------------------------------------------------

## 📜 License

MIT License. See `LICENSE` file for details.
