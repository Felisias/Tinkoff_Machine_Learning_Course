```python
import numpy as np

def bootstrap_sample(X, y, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    return X[indices], y[indices]
```