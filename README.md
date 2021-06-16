# deep-trainer

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


## Build and Deploy

```bash
$ pip install build twine
$ python -m build
$ python -m twine upload dist/*
```
