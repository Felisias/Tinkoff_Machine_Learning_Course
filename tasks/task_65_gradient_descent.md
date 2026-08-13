```python
import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    for _ in range(epochs):
        y_pred = X @ w
        grad = -2 * X.T @ (y - y_pred) / n_samples
        w -= lr * grad
        
    return w
```