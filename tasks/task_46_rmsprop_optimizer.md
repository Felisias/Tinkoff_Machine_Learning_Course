```python
import numpy as np

class RMSpropOptimizer:
    def __init__(self, lr=0.01, decay_rate=0.9, epsilon=1e-8):
        self.lr = lr
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.v = None

    def update(self, w, grad):
        if self.v is None:
            self.v = np.zeros_like(w)
            
        self.v = self.decay_rate * self.v + (1 - self.decay_rate) * (grad ** 2)
        w -= self.lr * grad / (np.sqrt(self.v) + self.epsilon)
        return w
```