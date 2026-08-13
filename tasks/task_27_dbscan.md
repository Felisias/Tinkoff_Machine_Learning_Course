```python
import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)
        cluster_id = 0
        distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        
        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue
                
            neighbors = np.where(distances[i] <= self.eps)[0]
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                continue
                
            self.labels_[i] = cluster_id
            seed_set = list(neighbors)
            seed_set.remove(i)
            
            while seed_set:
                q = seed_set.pop(0)
                if self.labels_[q] == -1:
                    self.labels_[q] = cluster_id
                if self.labels_[q] != -1:
                    continue
                    
                self.labels_[q] = cluster_id
                q_neighbors = np.where(distances[q] <= self.eps)[0]
                if len(q_neighbors) >= self.min_samples:
                    seed_set.extend(q_neighbors)
                    
            cluster_id += 1
        return self
```