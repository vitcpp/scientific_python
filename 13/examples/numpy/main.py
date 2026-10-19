import astromath
import numpy
import time

d = numpy.array(list(range(100_000_000)), dtype = numpy.int64)

tbeg = time.time()
print(numpy.sum(d))
tend = time.time()
print(f"time: {tend - tbeg}")

tbeg = time.time()
print(astromath.sum(d))
tend = time.time()
print(f"time: {tend - tbeg}")
