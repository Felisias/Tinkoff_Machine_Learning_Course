```python
import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.W = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            W_old = self.W.copy()
            for j in range(n_features):
                X_j = X[:, j]
                y_pred = X @ self.W - X_j * self.W[j]
                rho = X_j.T @ (y - y_pred)
                
                if rho < -self.alpha * n_samples:
                    self.W[j] = (rho + self.alpha * n_samples) / (X_j.T @ X_j)
                elif rho > self.alpha * n_samples:
                    self.W[j] = (rho - self.alpha * n_samples) / (X_j.T @ X_j)
                else:
                    self.W[j] = 0.0
                    
            if np.max(np.abs(self.W - W_old)) < self.tol:
                break
        return self

    def predict(self, X):
        return X @ self.W
```