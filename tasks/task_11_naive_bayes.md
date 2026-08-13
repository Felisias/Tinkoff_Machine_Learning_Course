```python
import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c])) - 0.5 * np.sum(((x - self.mean[c]) ** 2) / self.var[c])
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
```