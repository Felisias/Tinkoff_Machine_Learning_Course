```python
import numpy as np

class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X):
        return self

    def transform(self, X):
        n_samples, n_features = X.shape
        out = [np.ones((n_samples, 1)), X]
        
        if self.degree >= 2:
            for d in range(2, self.degree + 1):
                out.append(X ** d)
                
        return np.hstack(out)

    def fit_transform(self, X):
        return self.transform(X)
```