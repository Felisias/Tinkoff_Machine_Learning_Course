```python
import numpy as np

class Normalizer:
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, X):
        return self.transform(X)
```