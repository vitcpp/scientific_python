from astromath import sum as amsum
import time

#id(simple)
#dir(simple)
#help(simple)

l = list(range(100_000_000))

tbeg = time.time()
r = amsum(l)
tend = time.time()
print(f"time: {tend - tbeg}")
