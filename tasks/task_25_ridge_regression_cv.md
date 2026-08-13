```python
import numpy as np

class RidgeCV:
    def __init__(self, alphas=(0.1, 1.0, 10.0)):
        self.alphas = alphas
        self.best_alpha_ = None
        self.W = None

    def fit(self, X, y):
        best_loss = float('inf')
        n_features = X.shape[1]
        I = np.eye(n_features)
        
        for alpha in self.alphas:
            W = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
            preds = X @ W
            loss = np.mean((y - preds) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                self.best_alpha_ = alpha
                self.W = W
                
        return self

    def predict(self, X):
        return X @ self.W
```