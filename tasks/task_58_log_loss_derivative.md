```python
import numpy as np

def log_loss_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
```