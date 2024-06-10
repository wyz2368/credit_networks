import numpy as np

a = np.array([[1,2,3,10],
              [1,2,3,4],
              [1,2,3,4],
              [2,10,10,4]])

c = np.array([-1, -2, -1, -2])

b = a * c
print(np.shape(b))
print(b)
print(np.argmax(np.squeeze(b)))

print(np.argmax(a, axis=0))