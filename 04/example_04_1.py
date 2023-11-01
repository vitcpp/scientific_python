import os
import re

def scan_dir(rootpath):
    # Сканирование директории в поисках файлов в формате SBIG ST-6
    for dirpath, _, filenames in os.walk(rootpath):
        for fn in filter(lambda x: "SBIG" in x, filenames):
            yield os.path.join(dirpath, fn)

def sbig_read_headers(fd):
    # Чтение метаданных из SBIG файла
    for line in map(str.strip, fd):
        if "End" in line:
            return
        m = re.search("^([^=]*) = (.*)$", line)
        if m:
            yield m.group(1), m.group(2)

def scan_sbig_files(rootpath):
    # Рекурсивный поиск файлов и чтение метаданных
    for fn in scan_dir(rootpath):
        headers = {}
        with open(fn, mode="rt", errors="ignore", newline="\n") as fd:
            for k, v in sbig_read_headers(fd):
                headers[k] = v
        yield fn, headers

for fn, headers in scan_sbig_files("/s/science/ss433"):
    print(fn)
    for k, v in headers.items():
        print("\t{} : {}".format(k, v))

