import numpy as np

def test(a):
    a[0] = 1
    return a

a = np.zeros(5)
b = test(a)
print(a, b)