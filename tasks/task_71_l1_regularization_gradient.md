```python
import numpy as np

def l1_regularization_gradient(w, alpha=1.0):
    return alpha * np.sign(w)
```