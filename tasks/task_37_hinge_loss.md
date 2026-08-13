```python
import numpy as np

def hinge_loss(y_true, y_pred):
    y_true_mapped = np.where(y_true == 0, -1, 1)
    margins = 1 - y_true_mapped * y_pred
    return np.mean(np.maximum(0, margins))
```