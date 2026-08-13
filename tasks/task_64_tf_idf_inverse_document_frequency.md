```python
import numpy as np

def inverse_document_frequency(documents, term):
    n_docs = len(documents)
    df = sum(1 for doc in documents if term in doc)
    return np.log((n_docs + 1) / (df + 1)) + 1.0
```