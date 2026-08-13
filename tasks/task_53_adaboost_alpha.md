```python
import numpy as np

def calculate_adaboost_alpha(error_rate):
    error_rate = np.clip(error_rate, 1e-10, 1 - 1e-10)
    return 0.5 * np.log((1 - error_rate) / error_rate)
```