import struct

with open("data.bin", "wb", newline = None) as fd:
    b = struct.pack(">II", 1, 2)
    print(b)
    fd.write(b)

with open("data.bin", "rb") as fd:
    while True:
        b = fd.read(8)
        if len(b) == 0:
            break
        print(b)
        x, y = struct.unpack(">II", b)
        print(x, y)

