```python
import numpy as np

def max_pooling_1d(X, pool_size=2, stride=2):
    out_len = (len(X) - pool_size) // stride + 1
    return np.array([np.max(X[i*stride : i*stride+pool_size]) for i in range(out_len)])
```