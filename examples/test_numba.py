from numba import jit
from math import sqrt

def sum(n):
    total = 0
    for n in range(1, n):
        total += sqrt(n)
    return total

sum(100_000_000)
