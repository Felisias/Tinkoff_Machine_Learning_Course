```python
import numpy as np
from sklearn.base import ClassifierMixin

class Perceptron(ClassifierMixin):
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.W = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        y_ = np.where(y == 0, -1, 1)
        
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.W) + self.b
                if y_[idx] * linear_output <= 0:
                    self.W += self.lr * y_[idx] * x_i
                    self.b += self.lr * y_[idx]
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.W) + self.b
        return np.where(linear_output >= 0, 1, 0)
```