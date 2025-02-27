import numpy as np

def sum_rec(arr: list) -> float:
    if (len(arr) == 0):
        return 0
    elif (len(arr) == 1):
        return arr[0]
    return arr[0] + sum_rec(arr[1:])

arr = np.linspace(1, 1000, num=100)
print(sum_rec(arr), *arr)