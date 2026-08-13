```python
import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self._class_mapping = {}

    def fit(self, y):
        self.classes_ = np.unique(y)
        self._class_mapping = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self._class_mapping[c] for c in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)
```