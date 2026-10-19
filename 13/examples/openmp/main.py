from astromath import sum as amsum
import time

tbeg = time.time()
r = amsum(500_000_000)
tend = time.time()
print(f"time: {tend - tbeg}")
print(r)
