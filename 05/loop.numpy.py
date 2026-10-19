import numpy as np

vmin = float(2**53 - 1)
vmax = vmin + 10.0
for v in np.arange(vmin, vmax, 1.0):
    print(v)

