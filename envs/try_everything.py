import numpy as np

# a = np.array([[1,2,3,10],
#               [1,2,3,4],
#               [0,0, 1,0],
#               [2,10,10,4]])
#
# b = np.copy(a)
# print(b)
# b[0,0] = 10
#
# print(a)
# print(b)


a = np.array([[1,2,3,10],
              [1,2,3,4],
              [0,0, 1,0],
              [2,10,10,4]], dtype=float)

print(np.sum(a, axis=0))