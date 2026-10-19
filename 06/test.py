import timeit
import numpy as np

npdata = np.array(np.arange(0, 1000000.0))
listdata = list(np.arange(0, 1000000.0))

def sum():
    s = 0;
    for n in listdata:
        s += n
    return s

print(np.sum(npdata))
print(sum())

t1 = timeit.timeit(lambda: np.sum(npdata), number = 100)
print(f"numpy time: {t1}")

t2 = timeit.timeit(lambda: sum() , number = 100)
print(f"python time: {t2}")

print(f"python time / numpy time: {t2 / t1}")
