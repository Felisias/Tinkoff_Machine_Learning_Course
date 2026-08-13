```python
import numpy as np
from sklearn.base import RegressorMixin

class SGDLinearRegression(RegressorMixin):
    def __init__(self, lr=1e-3, max_steps=1000, batch_size=64):
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.W = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_steps):
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]
            predictions = X_batch @ self.W
            gradient = -2 * X_batch.T @ (y_batch - predictions) / self.batch_size
            self.W -= self.lr * gradient
        return self

    def predict(self, X):
        return X @ self.W
```