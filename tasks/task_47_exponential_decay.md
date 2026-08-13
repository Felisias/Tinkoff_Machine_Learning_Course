```python
def exponential_decay(initial_lr, step, decay_rate):
    return initial_lr * (decay_rate ** step)
```