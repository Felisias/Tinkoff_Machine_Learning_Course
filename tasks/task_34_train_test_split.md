```python
import numpy as np

def train_test_split(X, y, test_size=0.25, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    test_samples = int(n_samples * test_size)
    
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```