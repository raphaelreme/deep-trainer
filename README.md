# deep-trainer

[![Lint and Test](https://github.com/raphaelreme/deep-trainer/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/deep-trainer/actions/workflows/tests.yml)

Baseline code to train deep neural networks.
Currently only available for PyTorch Framework.


## Install

### Pip

```bash
$ pip install deep-trainer
```

### Conda

Not yet available


## Getting started

```python
import torch
from deep_trainer import PytorchTrainer


# Datasets
trainset = #....
valset = #....
testset = #....

# Dataloaders
train_loader = torch.utils.data.DataLoader(trainset, 64, shuffle=True)
val_loader = torch.data.utils.DataLoader(valset, 256)
test_loader = torch.data.utils.DataLoader(testset, 256)

# Model & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = #....
model.to(device)

# Optimizer & Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(trainset) * 50, 0.1)  # Decay by 10 every 50 epochs

# Criterion
criterion = torch.nn.CrossEntropyLoss()  # For classification for instance

# Training
trainer = PytorchTrainer(model, optimizer, scheduler, save_mode="small", device=device)
trainer.train(150, train_loader, criterion, val_loader=val_loader)

# Testing
trainer.load("experiments/checkpoints/best.ckpt")
trainer.evaluate(test_loader, criterion)
```


## Example

`example/example.py` shows how to train a PreActResNet with Deep Trainer.

Install the additional requirements and use it with:

```bash
$ # See hyperparameters available
$ python example.py -h
$
$ # Launch the default training
$ python example.py
$
$ # Once done (or during the training), look for default tensorboard logs
$ tensorboard --logdir experiments/logs/
```

This script is reaching around 94-95% accuracy on validation with Cifar10 and a PreActResNet18.
