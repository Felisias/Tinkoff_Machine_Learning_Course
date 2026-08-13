```python
import numpy as np

def cosine_similarity(X, Y):
    dot_product = X @ Y.T
    norm_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
    norm_Y = np.linalg.norm(Y, axis=1)[np.newaxis, :]
    return dot_product / (norm_X * norm_Y)
```