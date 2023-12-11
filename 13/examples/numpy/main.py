import astromath
import numpy

d = numpy.array(list(range(100)), dtype = numpy.int64)
print(numpy.sum(d))
print(astromath.sum(d))
