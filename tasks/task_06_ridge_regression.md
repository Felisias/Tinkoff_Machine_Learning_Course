```python
import numpy as np
from sklearn.base import RegressorMixin

class RidgeRegression(RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.W = None

    def fit(self, X, y):
        n_features = X.shape[1]
        I = np.eye(n_features)
        self.W = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y
        return self

    def predict(self, X):
        return X @ self.W
```