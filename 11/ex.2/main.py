import numpy

def calc_sum():
    sum = 0
    for x in range(100_000_000):
        sum += x
    return sum

def calc_sum_numpy():
    values = numpy.arange(0, 100_000_000, dtype=numpy.int64)
    return numpy.sum(values)

def calc_sub():
    res = 0
    for x in range(10_000_000):
        res -= x
    return res

def calc_mul():
    res = 0
    for x in range(10_000_000):
        res *= x
    return res

def main():
    calc_sum()
    calc_sum_numpy()
    calc_sub()
    calc_mul()

main()
