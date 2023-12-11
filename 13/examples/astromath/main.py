from astromath import sum as amsum
import time

#id(simple)
#dir(simple)
#help(simple)

print(amsum(2, 2))
print(amsum([10, 20, 4, 5, 6]))
print(amsum(range(100)))

l = list(range(100_000_000))

tbeg = time.time()
#r = amsum(range(100_000_000))
r = amsum(l)
tend = time.time()
print(f"time: {tend - tbeg}")
