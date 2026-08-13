```python
import numpy as np
from sklearn.base import ClassifierMixin

class SGDLogisticRegression(ClassifierMixin):
    def __init__(self, lr=1e-3, max_steps=1000, batch_size=64):
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.W = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_steps):
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]
            predictions = self._sigmoid(X_batch @ self.W)
            gradient = X_batch.T @ (predictions - y_batch) / self.batch_size
            self.W -= self.lr * gradient
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.W)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
```