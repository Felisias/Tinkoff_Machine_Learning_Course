```python
import numpy as np

def simple_moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size
```