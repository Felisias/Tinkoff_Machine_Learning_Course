```python
import numpy as np

def roc_auc_score(y_true, y_score):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    return np.trapz(tpr, fpr)
```