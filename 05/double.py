#!/usr/bin/python3

import math
import numpy

def array_sum():
    result = 0.0;
    v = 0
    while v < 1000:
        result += v
        v += 0.1
    return result;

def printd(e : str):
    print("{:<50} = {:.18f}".format(e, eval(e)))

print("-------- native python --------")
printd("1.125 + 1.5")
printd("1.1 + 1.2")
printd("1.2 - 1.1")
printd("1.23254332482E-12 + 3.9382498452394832094E+9")
printd("math.sin(0.237238493)");
printd("math.cos(3.9382498452394832094E+9)")
printd("math.log10(3.9382498452394832094E+9)")
printd("array_sum()")
printd("math.nan")

print("-------- numpy --------")
printd("numpy.sin(0.237238493)");
printd("numpy.cos(3.9382498452394832094E+9)")
printd("numpy.log10(3.9382498452394832094E+9)")
