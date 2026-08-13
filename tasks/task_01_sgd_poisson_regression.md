```python
import numpy as np
from sklearn.base import RegressorMixin

class SGDPoissonRegression(RegressorMixin):
    def __init__(self, lr=1e-3, momentum=1, delta_converged=1e-3, max_steps=1000, batch_size=64):
        self.lr = lr
        self.momentum = momentum
        self.delta_converged = delta_converged
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.W = None

    def fit(self, X, y):
        np.random.seed(42)
        n_samples, n_features = X.shape
        self.W = np.random.randn(n_features)
        vt = np.zeros(n_features)
        
        for _ in range(self.max_steps):
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            linear_pred = X_batch @ self.W
            predictions = np.exp(linear_pred)
            
            gradient = X_batch.T @ (predictions - y_batch) / self.batch_size
            
            vt = self.momentum * vt - self.lr * gradient
            W_new = self.W + vt
            
            if np.linalg.norm(W_new - self.W) < self.delta_converged:
                self.W = W_new
                break
                
            self.W = W_new
        return self

    def predict(self, X):
        return np.exp(X @ self.W)
```