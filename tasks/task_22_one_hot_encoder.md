```python
import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        out = []
        for i in range(X.shape[1]):
            col = X[:, i]
            cats = self.categories_[i]
            encoded = np.zeros((len(col), len(cats)))
            for j, val in enumerate(col):
                if val in cats:
                    encoded[j, np.where(cats == val)[0][0]] = 1
            out.append(encoded)
        return np.hstack(out)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```