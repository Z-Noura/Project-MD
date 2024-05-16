import numpy as np

arrays = [100*np.random.randn(1, 2) for _ in range(10)]
print(arrays)
new = np.stack(arrays, axis=0)
print(new)