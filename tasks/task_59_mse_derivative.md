```python
import numpy as np

def mse_derivative(y_true, y_pred):
    n = len(y_true)
    return -2 * (y_true - y_pred) / n
```