```python
import numpy as np

def jaccard_score(y_true, y_pred):
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    
    if union == 0:
        return 0.0
    return intersection / union
```