```python
import numpy as np

class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variances_ = None
        self.features_to_keep_ = None

    def fit(self, X):
        self.variances_ = np.var(X, axis=0)
        self.features_to_keep_ = self.variances_ > self.threshold
        return self

    def transform(self, X):
        return X[:, self.features_to_keep_]

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```