```python
import numpy as np

class SimpleImputer:
    def __init__(self):
        self.statistics_ = None

    def fit(self, X):
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for i in range(X.shape[1]):
            nan_mask = np.isnan(X_transformed[:, i])
            X_transformed[nan_mask, i] = self.statistics_[i]
        return X_transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```