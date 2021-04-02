import numpy as np

# arr = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
a = np.array([[1, 2], [2, 3], [3, 4]])
print(a.shape)
b = [0, 1, 1]
print(a[b, : -1])