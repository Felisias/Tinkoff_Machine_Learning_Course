```python
import numpy as np

def dropout_forward(X, drop_prob=0.5):
    keep_prob = 1.0 - drop_prob
    mask = np.random.binomial(1, keep_prob, size=X.shape) / keep_prob
    return X * mask
```