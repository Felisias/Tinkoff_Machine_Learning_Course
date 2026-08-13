```python
import numpy as np

def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))
```