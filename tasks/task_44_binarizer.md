```python
import numpy as np

class Binarizer:
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X):
        return self

    def transform(self, X):
        return np.where(X > self.threshold, 1.0, 0.0)

    def fit_transform(self, X):
        return self.transform(X)
```