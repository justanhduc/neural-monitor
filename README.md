# Neural Monitor

Do your training. We will take care of the statistics.

## Installation

```
pip install git+https://github.com/justanhduc/neural-monitor
```

## Usages

The basic usage in most cases will be

```
from neural_monitor import monitor as mon

# Tensorboard is turned on by default
mon.initialize(model_name='foo-model', print_freq=100, use_tensorboard=True)
...

def calculate_loss(pred, gt):
    ...
    training_loss = ...
    mon.plot('training loss', loss, smooth=.99, filter_outliers=True)

def calculate_acc(pred, gt):
    accuracy = ...
    mon.plot('training acc', accuracy, smooth=.99, filter_outliers=True)

...
for epoch in mon.iter_epoch(range(n_epochs)):
    for data in mon.iter_batch(data_loader):
        pred = net(data)
        calculate_loss(pred, gt)
        calculate_acc(pred, gt)
        mon.imwrite('input images', data['images'], latest_only=True)

    mon.dump('checkpoint.pt', {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        ...
    }, method='torch', keep=5)  # keep only 5 latest checkpoints
...

```
For more details on Neural Monitor's functionality, please check the [documentation](https://neuralnet-pytorch.readthedocs.io/en/latest/).

## References

This project is inspired by [WGAN-GP](https://github.com/igul222/improved_wgan_training).

## Related repos

[Neuralnet-pytorch](https://github.com/justanhduc/neuralnet-pytorch)
