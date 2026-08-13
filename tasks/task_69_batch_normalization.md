```python
import numpy as np

def batch_norm_forward(X, gamma=1.0, beta=0.0, eps=1e-5):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mu) / np.sqrt(var + eps)
    return gamma * X_norm + beta
```