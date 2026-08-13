```python
import numpy as np

def term_frequency(document):
    words = document.split()
    tf_dict = {}
    total_words = len(words)
    
    for word in words:
        tf_dict[word] = tf_dict.get(word, 0) + 1
        
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / total_words
        
    return tf_dict
```